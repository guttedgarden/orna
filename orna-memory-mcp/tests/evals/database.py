"""Изолированная PostgreSQL database и fixture loader для retrieval eval."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from ipaddress import ip_address
from uuid import uuid4

import asyncpg

from app.config import Settings
from app.db import create_db_pool, run_database_migrations
from app.embeddings import AsyncEmbeddingBackend
from app.models import MemoryInsertRecord, MemoryScope, MemoryStatus
from app.normalizer import build_lexical_source, canonical_content_hash
from tests.evals.dataset import CorpusRecord

EVAL_DATABASE_PREFIX = "orna_eval_"
_SAFE_DATABASE_NAME = re.compile(r"^orna_eval_[a-z0-9_]+$")


class EvalDatabaseError(RuntimeError):
    """Нарушен safety contract изолированной eval database."""


def _require_loopback_host(host: str) -> None:
    """Не разрешает eval DDL на remote PostgreSQL server из обычного `.env`."""

    normalized = host.strip().removeprefix("[").removesuffix("]")
    if normalized == "localhost":
        return
    try:
        is_loopback = ip_address(normalized).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise EvalDatabaseError("eval database admin host must be an explicit loopback address")


@dataclass(frozen=True, slots=True)
class EvalDatabase:
    """Ресурсы одной ephemeral database, принадлежащей текущему run."""

    name: str
    settings: Settings
    pool: asyncpg.Pool


def derive_eval_settings(base: Settings, database_name: str) -> Settings:
    """Заменяет любой configured target на явно созданную eval database."""

    _require_loopback_host(base.postgres_host)
    if not _SAFE_DATABASE_NAME.fullmatch(database_name):
        raise EvalDatabaseError(f"eval database name must use {EVAL_DATABASE_PREFIX} prefix")
    values = base.model_dump(exclude={"database_url", "postgres_db"})
    return Settings(
        **values,
        postgres_db=database_name,
        database_url=None,
        _env_file=None,
    )


async def _drop_owned_database(admin: asyncpg.Connection, database_name: str) -> None:
    if not _SAFE_DATABASE_NAME.fullmatch(database_name):
        raise EvalDatabaseError("refusing to drop a database not owned by this eval run")
    await admin.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = $1 AND pid <> pg_backend_pid();",
        database_name,
    )
    await admin.execute(f'DROP DATABASE "{database_name}";')


@asynccontextmanager
async def ephemeral_eval_database(base: Settings) -> AsyncIterator[EvalDatabase]:
    """Создаёт, мигрирует и гарантированно удаляет только свою eval database."""

    database_name = f"{EVAL_DATABASE_PREFIX}{uuid4().hex[:12]}"
    _require_loopback_host(base.postgres_host)
    admin = await asyncpg.connect(
        host=base.postgres_host,
        port=base.postgres_port,
        user=base.postgres_user,
        password=base.postgres_password,
        database="template1",
    )
    created = False
    pool: asyncpg.Pool | None = None
    try:
        await admin.execute(f'CREATE DATABASE "{database_name}" TEMPLATE template0;')
        created = True
        settings = derive_eval_settings(base, database_name)
        await run_database_migrations(settings)
        pool = await create_db_pool(settings)
        yield EvalDatabase(name=database_name, settings=settings, pool=pool)
    finally:
        try:
            if pool is not None:
                await pool.close()
            if created:
                await _drop_owned_database(admin, database_name)
        finally:
            await admin.close()


async def prepare_corpus_records(
    corpus: Sequence[CorpusRecord],
    embeddings: AsyncEmbeddingBackend,
    settings: Settings,
) -> tuple[MemoryInsertRecord, ...]:
    """Применяет реальные hash/lexical/embedding preparation paths к fixed fixtures."""

    prepared: list[MemoryInsertRecord] = []
    for fixture in corpus:
        prepared.append(
            MemoryInsertRecord(
                id=fixture.id,
                logical_id=fixture.logical_id,
                revision=fixture.revision,
                supersedes_id=fixture.supersedes_id,
                scope=MemoryScope(fixture.scope),
                project_id=fixture.project_id,
                memory_type=fixture.memory_type,
                status=MemoryStatus(fixture.status),
                content=fixture.content,
                content_hash=canonical_content_hash(fixture.content),
                tags=list(fixture.tags),
                identifiers=list(fixture.identifiers),
                lexical_source=build_lexical_source(
                    fixture.content,
                    fixture.tags,
                    fixture.identifiers,
                ),
                lexical_profile_version=settings.lexical_profile_version,
                embedding=await embeddings.embed_memory(fixture.content),
                embedding_model=settings.embedding_model,
                embedding_profile_version=settings.embedding_profile_version,
                provenance={
                    "source": {
                        "kind": "eval_fixture",
                        "source_ref": fixture.source_ref,
                    }
                },
            )
        )
    return tuple(prepared)


async def load_eval_corpus(
    pool: asyncpg.Pool,
    corpus: Sequence[CorpusRecord],
    embeddings: AsyncEmbeddingBackend,
    settings: Settings,
) -> None:
    """Вставляет fixed revisions в одной test-fixture transaction."""

    prepared = await prepare_corpus_records(corpus, embeddings, settings)
    ordered = sorted(prepared, key=lambda record: (record.logical_id.int, record.revision))
    async with pool.acquire() as conn, conn.transaction():
        for record in ordered:
            await conn.execute(
                """
                INSERT INTO memories (
                    id, logical_id, revision, supersedes_id, scope, project_id,
                    memory_type, status, content, content_hash, tags, identifiers,
                    lexical_source, lexical_profile_version, embedding, embedding_model,
                    embedding_profile_version, provenance
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18::jsonb
                );
                """,
                record.id,
                record.logical_id,
                record.revision,
                record.supersedes_id,
                record.scope.value,
                record.project_id,
                record.memory_type,
                record.status.value,
                record.content,
                record.content_hash,
                record.tags,
                record.identifiers,
                record.lexical_source,
                record.lexical_profile_version,
                record.embedding,
                record.embedding_model,
                record.embedding_profile_version,
                json.dumps(record.provenance),
            )


async def inspect_eval_database(pool: asyncpg.Pool) -> tuple[dict[str, str], dict[str, int]]:
    """Возвращает безопасные версии и размеры без DSN или содержимого records."""

    async with pool.acquire() as conn:
        version = await conn.fetchval("SHOW server_version;")
        pgvector = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
        )
        records = await conn.fetchval("SELECT count(*) FROM memories;")
        index_bytes = await conn.fetchval(
            "SELECT COALESCE(sum(pg_relation_size(indexrelid)), 0) "
            "FROM pg_index WHERE indrelid = 'memories'::regclass;"
        )
    return (
        {"postgresql_version": str(version), "pgvector_version": str(pgvector)},
        {"records": int(records), "index_bytes": int(index_bytes)},
    )
