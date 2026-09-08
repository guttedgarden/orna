"""Интеграционный тест application services поверх реального PostgreSQL."""

import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from app.config import Settings
from app.db import init_connection, run_database_migrations
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE
from app.embeddings import AsyncEmbeddingExecutor, EmbeddingService
from app.models import EMBEDDING_DIMENSION, MemoryScope
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService
from app.write import MemoryAddCommand, MemoryWriteService
from app.write_safety import E5LengthGuard, MemoryWriteSafety


class DeterministicEmbeddings:
    """Заменяет дорогой E5 inference, не подменяя storage/retrieval pipeline."""

    @staticmethod
    async def embed_memory(content: str) -> list[float]:
        index = 0 if "MariaDB" in content else 1
        vector = [0.0] * EMBEDDING_DIMENSION
        vector[index] = 1.0
        return vector

    @staticmethod
    async def embed_query(_query: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSION
        vector[0] = 1.0
        return vector


class AllowAllSafety:
    """Application integration double; real tokenizer contract has dedicated tests."""

    @staticmethod
    def validate(**_values: object) -> None:
        return None


def _required_real_e5_cache_dir() -> Path:
    configured = os.environ.get("ORNA_TEST_E5_CACHE_DIR")
    cache_dir = Path(configured) if configured else Settings(_env_file=None).embedding_cache_dir
    snapshot = ACTIVE_EMBEDDING_PROFILE.snapshot_path(cache_dir)
    if not snapshot.is_dir():
        raise AssertionError(
            "pinned E5 cache unavailable; run the documented model-cache workflow and set "
            "ORNA_TEST_E5_CACHE_DIR"
        )
    return cache_dir


@pytest.fixture
async def service_database() -> AsyncIterator[tuple[Settings, asyncpg.Pool]]:
    base_settings = Settings()
    database_name = f"orna_service_test_{uuid4().hex[:10]}"
    admin_conn = await asyncpg.connect(
        f"postgresql://{base_settings.postgres_user}:{base_settings.postgres_password}"
        f"@{base_settings.postgres_host}:{base_settings.postgres_port}/template1"
    )
    await admin_conn.execute(f'CREATE DATABASE "{database_name}";')
    test_settings = base_settings.model_copy(
        update={
            "postgres_db": database_name,
            "database_url": (
                f"postgresql://{base_settings.postgres_user}:{base_settings.postgres_password}"
                f"@{base_settings.postgres_host}:{base_settings.postgres_port}/{database_name}"
            ),
        }
    )
    await run_database_migrations(test_settings)
    pool = await asyncpg.create_pool(
        dsn=test_settings.database_url,
        min_size=2,
        max_size=2,
        init=init_connection,
    )
    try:
        yield test_settings, pool
    finally:
        await pool.close()
        await admin_conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = $1 AND pid <> pg_backend_pid();",
            database_name,
        )
        await admin_conn.execute(f'DROP DATABASE "{database_name}";')
        await admin_conn.close()


async def test_write_and_hybrid_search_preserve_project_isolation(
    service_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = service_database
    repository = MemoryRepository(pool, settings)
    embeddings = DeterministicEmbeddings()
    writer = MemoryWriteService(repository, embeddings, settings, AllowAllSafety())
    searcher = MemorySearchService(repository, embeddings, settings)
    legacy_memory = await writer.add(
        MemoryAddCommand(
            content="Migration tests use MariaDB.",
            scope=MemoryScope.PROJECT,
            memory_type="decision",
            identifiers=["MigrationRunner"],
        ),
        project_id="legacy-api",
    )
    orna_memory = await writer.add(
        MemoryAddCommand(
            content="Migration tests use PostgreSQL.",
            scope=MemoryScope.PROJECT,
            memory_type="decision",
        ),
        project_id="orna-memory",
    )
    global_memory = await writer.add(
        MemoryAddCommand(
            content="Keep migration logs for every project.",
            scope=MemoryScope.GLOBAL,
            memory_type="decision",
        ),
        project_id="legacy-api",
    )
    await writer.add(
        MemoryAddCommand(
            content="MariaDB is also mentioned in an unrelated fact.",
            scope=MemoryScope.PROJECT,
            memory_type="fact",
        ),
        project_id="legacy-api",
    )

    assert legacy_memory.id.version == 7
    assert legacy_memory.logical_id.version == 7

    results = await searcher.search(
        MemorySearchQuery(
            query="Which database is used for migration tests?",
            memory_type="decision",
            limit=5,
        ),
        project_id="legacy-api",
    )

    result_ids = [result.id for result in results]
    assert result_ids == [legacy_memory.id, global_memory.id]
    assert orna_memory.id not in result_ids
    assert results[0].rank_dense == 1
    assert results[0].rank_lexical is None


async def test_multilingual_query_finds_english_memory_with_real_pinned_e5(
    service_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = service_database
    settings = settings.model_copy(
        update={
            "embedding_cache_dir": _required_real_e5_cache_dir(),
            "embedding_local_files_only": True,
        }
    )
    repository = MemoryRepository(pool, settings)
    embeddings = AsyncEmbeddingExecutor(
        EmbeddingService(settings),
        max_concurrency=settings.embedding_max_concurrency,
    )
    writer = MemoryWriteService(
        repository,
        embeddings,
        settings,
        MemoryWriteSafety(E5LengthGuard(settings)),
    )
    searcher = MemorySearchService(repository, embeddings, settings)

    stored = await writer.add(
        MemoryAddCommand(
            content="Migration tests must run against MariaDB, not SQLite.",
            scope=MemoryScope.PROJECT,
            memory_type="convention",
        ),
        project_id="test-project",
    )
    distractor_contents = (
        "Frontend snapshot tests must run in Chromium, not Firefox.",
        "Background jobs retry transient failures three times.",
        "API response caches expire after fifteen minutes.",
        "Deployment artifacts must be signed before release.",
        "Application logs are retained for thirty days.",
    )
    for content in distractor_contents:
        await writer.add(
            MemoryAddCommand(
                content=content,
                scope=MemoryScope.PROJECT,
                memory_type="convention",
            ),
            project_id="test-project",
        )

    results = await searcher.search(
        MemorySearchQuery(
            query="На какой базе нужно запускать тесты миграций?",
            limit=5,
        ),
        project_id="test-project",
    )

    assert len(results) == 5
    assert results[0].id == stored.id
    stored_result = results[0]
    assert stored_result.rank_dense == 1
    assert stored_result.rank_lexical is None
