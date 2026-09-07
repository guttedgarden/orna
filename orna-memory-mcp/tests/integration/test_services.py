"""Интеграционный тест application services поверх реального PostgreSQL."""

from collections.abc import AsyncIterator
from uuid import uuid4

import asyncpg
import pytest

from app.config import Settings
from app.db import init_connection, run_database_migrations
from app.models import EMBEDDING_DIMENSION, MemoryScope
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService
from app.write import MemoryAddCommand, MemoryWriteService


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
    writer = MemoryWriteService(repository, embeddings, settings)
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
