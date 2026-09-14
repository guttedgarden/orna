"""Интеграционные тесты PostgreSQL repository и retrieval-каналов."""

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import asyncpg
import pytest

from app.config import Settings
from app.db import init_connection, run_database_migrations
from app.models import EMBEDDING_DIMENSION, MemoryInsertRecord, MemoryScope, MemoryStatus
from app.normalizer import (
    build_lexical_source,
    canonical_content_hash,
    normalize_query_to_lexical_groups,
)
from app.repository import MemoryRepository


@pytest.fixture
async def repository_database() -> AsyncIterator[tuple[Settings, asyncpg.Pool]]:
    """Создаёт отдельную БД и single-connection pool для наблюдения за GUC isolation."""
    base_settings = Settings()
    database_name = f"orna_repository_test_{uuid4().hex[:10]}"
    admin_conn = await asyncpg.connect(
        host=base_settings.postgres_host,
        port=base_settings.postgres_port,
        user=base_settings.postgres_user,
        password=base_settings.postgres_password,
        database="template1",
    )
    await admin_conn.execute(f'CREATE DATABASE "{database_name}" TEMPLATE template0;')

    test_settings = Settings(
        **base_settings.model_dump(
            exclude={
                "database_url",
                "postgres_db",
                "hnsw_ef_search",
                "hnsw_iterative_scan",
            }
        ),
        postgres_db=database_name,
        hnsw_ef_search=73,
        hnsw_iterative_scan="strict_order",
        database_url=None,
    )

    await run_database_migrations(test_settings)
    pool = await asyncpg.create_pool(
        dsn=test_settings.database_url,
        min_size=1,
        max_size=1,
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


def unit_vector(primary_index: int, secondary_index: int | None = None) -> list[float]:
    """Строит ненулевой deterministic vector для cosine distance."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[primary_index] = 1.0
    if secondary_index is not None:
        vector[secondary_index] = 0.1
    return vector


def memory_record(
    *,
    content: str,
    embedding: list[float],
    record_id: UUID | None = None,
    scope: MemoryScope = MemoryScope.GLOBAL,
    project_id: str | None = None,
    memory_type: str = "fact",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    identifiers: list[str] | None = None,
) -> MemoryInsertRecord:
    """Создаёт валидный insert DTO с реальными lexical/hash representations."""
    record_identifiers = identifiers or []
    return MemoryInsertRecord(
        id=uuid4() if record_id is None else record_id,
        logical_id=uuid4(),
        revision=1,
        supersedes_id=None,
        scope=scope,
        project_id=project_id,
        memory_type=memory_type,
        status=status,
        content=content,
        content_hash=canonical_content_hash(content),
        tags=["repository"],
        identifiers=record_identifiers,
        lexical_source=build_lexical_source(content, ["repository"], record_identifiers),
        lexical_profile_version="lexical-v1",
        embedding=embedding,
        embedding_model="intfloat/multilingual-e5-large",
        embedding_profile_version="e5-v1",
        provenance={"source": "integration-test"},
    )


async def test_insert_and_get_hydrate_complete_records(
    repository_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = repository_database
    repository = MemoryRepository(pool, settings)
    inserted = await repository.insert(
        memory_record(content="Repository hydration", embedding=unit_vector(0))
    )

    assert inserted.provenance == {"source": "integration-test"}
    assert inserted.created_at is not None
    assert await repository.get_by_id(inserted.id, "project-a") == inserted
    assert await repository.get_active_by_logical_id(inserted.logical_id, "project-a") == inserted
    assert await repository.get_by_id(uuid4(), "project-a") is None
    assert await repository.get_active_by_logical_id(uuid4(), "project-a") is None


async def test_get_lookups_filter_project_visibility(
    repository_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = repository_database
    repository = MemoryRepository(pool, settings)
    project_a = await repository.insert(
        memory_record(
            content="Project A lookup",
            embedding=unit_vector(0),
            scope=MemoryScope.PROJECT,
            project_id="project-a",
        )
    )
    project_b = await repository.insert(
        memory_record(
            content="Project B lookup",
            embedding=unit_vector(1),
            scope=MemoryScope.PROJECT,
            project_id="project-b",
        )
    )
    global_memory = await repository.insert(
        memory_record(content="Global lookup", embedding=unit_vector(2))
    )

    assert await repository.get_by_id(project_a.id, "project-a") == project_a
    assert await repository.get_by_id(project_a.id, "project-b") is None
    assert await repository.get_by_id(project_b.id, "project-a") is None
    assert await repository.get_by_id(global_memory.id, "project-a") == global_memory

    assert await repository.get_active_by_logical_id(project_a.logical_id, "project-a") == project_a
    assert await repository.get_active_by_logical_id(project_a.logical_id, "project-b") is None
    assert await repository.get_active_by_logical_id(project_b.logical_id, "project-a") is None
    assert (
        await repository.get_active_by_logical_id(global_memory.logical_id, "project-a")
        == global_memory
    )


@pytest.mark.parametrize("strategy", ["exact", "hnsw"])
async def test_search_dense_filters_status_and_project_visibility(
    repository_database: tuple[Settings, asyncpg.Pool],
    strategy: str,
) -> None:
    settings, pool = repository_database
    repository = MemoryRepository(pool, settings)

    project_a = await repository.insert(
        memory_record(
            content="Project A",
            embedding=unit_vector(0),
            scope=MemoryScope.PROJECT,
            project_id="project-a",
        )
    )
    global_memory = await repository.insert(
        memory_record(content="Global", embedding=unit_vector(0, 1))
    )
    hidden_records = [
        await repository.insert(
            memory_record(
                content="Project B",
                embedding=unit_vector(0),
                scope=MemoryScope.PROJECT,
                project_id="project-b",
            )
        ),
        await repository.insert(
            memory_record(
                content="Superseded",
                embedding=unit_vector(0),
                status=MemoryStatus.SUPERSEDED,
            )
        ),
        await repository.insert(
            memory_record(
                content="Archived",
                embedding=unit_vector(0),
                status=MemoryStatus.ARCHIVED,
            )
        ),
    ]

    if strategy == "hnsw":
        # Маленькая таблица обычно предпочитает seq scan; принуждаем planner проверить HNSW path.
        async with pool.acquire() as conn:
            await conn.execute("SET enable_seqscan = off")

    results = await repository.search_dense(unit_vector(0), "project-a", 10, strategy=strategy)
    result_ids = [record.id for record, _distance in results]

    assert result_ids == [project_a.id, global_memory.id]
    assert not ({record.id for record in hidden_records} & set(result_ids))
    assert results[0][1] == pytest.approx(0.0)

    # Repository search settings должны быть transaction-local и не загрязнять pool.
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT current_setting('enable_indexscan')") == "on"
        assert await conn.fetchval("SELECT current_setting('hnsw.ef_search')") == "40"
        assert await conn.fetchval("SELECT current_setting('hnsw.iterative_scan')") == "off"
        await conn.execute("SET enable_seqscan = on")


async def test_search_dense_uses_configured_strategy_when_override_is_absent(
    repository_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = repository_database
    hnsw_settings = settings.model_copy(update={"dense_retrieval_strategy": "hnsw"})
    repository = MemoryRepository(pool, hnsw_settings)
    expected = await repository.insert(
        memory_record(content="Configured HNSW", embedding=unit_vector(2))
    )

    results = await repository.search_dense(unit_vector(2), None, 1)

    assert [record.id for record, _distance in results] == [expected.id]


@pytest.mark.parametrize("strategy", ["exact", "hnsw"])
async def test_search_dense_stabilizes_equal_distance_ties(
    repository_database: tuple[Settings, asyncpg.Pool],
    strategy: str,
) -> None:
    settings, pool = repository_database
    repository = MemoryRepository(pool, settings)
    lower_id = UUID(int=1)
    higher_id = UUID(int=2)
    tied_vector = unit_vector(7)

    # Обратный insert order доказывает, что порядок не зависит от natural table order.
    await repository.insert(
        memory_record(content="Higher UUID", embedding=tied_vector, record_id=higher_id)
    )
    await repository.insert(
        memory_record(content="Lower UUID", embedding=tied_vector, record_id=lower_id)
    )

    if strategy == "hnsw":
        async with pool.acquire() as conn:
            await conn.execute("SET enable_seqscan = off")

    results = await repository.search_dense(tied_vector, None, 10, strategy=strategy)

    if strategy == "hnsw":
        async with pool.acquire() as conn:
            await conn.execute("SET enable_seqscan = on")

    assert [record.id for record, _distance in results] == [lower_id, higher_id]
    assert [distance for _record, distance in results] == pytest.approx([0.0, 0.0])


async def test_search_lexical_handles_identifiers_hyphens_and_visibility(
    repository_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = repository_database
    repository = MemoryRepository(pool, settings)
    executor = await repository.insert(
        memory_record(
            content="Executes upstream response providers.",
            embedding=unit_vector(3),
            identifiers=["ResponseProviderExecutor"],
        )
    )
    hyphenated = await repository.insert(
        memory_record(
            content="Propagate request metadata.",
            embedding=unit_vector(4),
            identifiers=["foo-bar", "x-request-id"],
        )
    )
    await repository.insert(
        memory_record(
            content="ResponseProviderExecutor is private.",
            embedding=unit_vector(5),
            scope=MemoryScope.PROJECT,
            project_id="project-b",
            identifiers=["ResponseProviderExecutor"],
        )
    )

    identifier_results = await repository.search_lexical(
        normalize_query_to_lexical_groups("ResponseProviderExecutor"),
        "project-a",
        10,
    )
    split_identifier_results = await repository.search_lexical(
        normalize_query_to_lexical_groups("response provider executor"),
        "project-a",
        10,
    )
    foo_bar_results = await repository.search_lexical(
        normalize_query_to_lexical_groups("foo-bar"), "project-a", 10
    )
    request_id_results = await repository.search_lexical(
        normalize_query_to_lexical_groups("x-request-id"), "project-a", 10
    )

    assert [record.id for record, _score in identifier_results] == [executor.id]
    assert [record.id for record, _score in split_identifier_results] == [executor.id]
    assert [record.id for record, _score in foo_bar_results] == [hyphenated.id]
    assert [record.id for record, _score in request_id_results] == [hyphenated.id]
    assert all(score > 0 for _record, score in identifier_results)


async def test_search_lexical_compound_groups_keep_alternatives_and_and_semantics(
    repository_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = repository_database
    repository = MemoryRepository(pool, settings)
    raw_compound = await repository.insert(
        memory_record(
            content="ResponseProviderExecutor timeout",
            embedding=unit_vector(6),
        )
    )
    expanded_compound = await repository.insert(
        memory_record(
            content="response provider executor timeout",
            embedding=unit_vector(7),
        )
    )
    without_timeout = await repository.insert(
        memory_record(
            content="ResponseProviderExecutor",
            embedding=unit_vector(8),
        )
    )
    explicitly_identified = await repository.insert(
        memory_record(
            content="Executor metadata is available.",
            embedding=unit_vector(9),
            identifiers=["ResponseProviderExecutor"],
        )
    )

    compound_results = await repository.search_lexical(
        normalize_query_to_lexical_groups("ResponseProviderExecutor timeout"),
        "project-a",
        10,
    )
    raw_identifier_results = await repository.search_lexical(
        normalize_query_to_lexical_groups("ResponseProviderExecutor"),
        "project-a",
        10,
    )
    split_identifier_results = await repository.search_lexical(
        normalize_query_to_lexical_groups("response provider executor"),
        "project-a",
        10,
    )

    compound_result_ids = {record.id for record, _score in compound_results}
    assert {raw_compound.id, expanded_compound.id} <= compound_result_ids
    assert without_timeout.id not in compound_result_ids
    assert explicitly_identified.id in {record.id for record, _score in raw_identifier_results}
    assert explicitly_identified.id in {record.id for record, _score in split_identifier_results}
    assert all(score > 0 for _record, score in compound_results)


async def test_search_lexical_normalizes_technical_inputs_and_filters_before_limit(
    repository_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = repository_database
    repository = MemoryRepository(pool, settings)
    visible = await repository.insert(
        memory_record(
            content="ResponseProviderExecutor timeout",
            embedding=unit_vector(10),
            record_id=UUID(int=10),
            scope=MemoryScope.PROJECT,
            project_id="project-a",
            memory_type="decision",
        )
    )
    for record_id, scope, project_id, memory_type, status in (
        (UUID(int=1), MemoryScope.PROJECT, "project-b", "decision", MemoryStatus.ACTIVE),
        (UUID(int=2), MemoryScope.PROJECT, "project-a", "decision", MemoryStatus.ARCHIVED),
        (UUID(int=3), MemoryScope.PROJECT, "project-a", "fact", MemoryStatus.ACTIVE),
    ):
        await repository.insert(
            memory_record(
                content="ResponseProviderExecutor timeout",
                embedding=unit_vector(11),
                record_id=record_id,
                scope=scope,
                project_id=project_id,
                memory_type=memory_type,
                status=status,
            )
        )

    filtered_results = await repository.search_lexical(
        normalize_query_to_lexical_groups("ResponseProviderExecutor timeout"),
        "project-a",
        1,
        memory_type="decision",
    )

    assert [record.id for record, _score in filtered_results] == [visible.id]

    for index, query in enumerate(
        (
            "getHTTPResponse",
            "routing_pool",
            "X-Memory-Project",
            "Application.php",
            "foo.bar",
            "namespace/ClassName",
            "550e8400-e29b-41d4-a716-446655440000 checksum",
            "МодульПамяти",
        ),
        start=12,
    ):
        stored = await repository.insert(memory_record(content=query, embedding=unit_vector(index)))
        results = await repository.search_lexical(
            normalize_query_to_lexical_groups(query), "project-a", 20
        )

        assert stored.id in {record.id for record, _score in results}

    assert await repository.search_lexical([], "project-a", 10) == []


async def test_get_by_id_returns_visible_non_active_record_but_logical_lookup_ignores_it(
    repository_database: tuple[Settings, asyncpg.Pool],
) -> None:
    settings, pool = repository_database
    repository = MemoryRepository(pool, settings)
    archived = await repository.insert(
        memory_record(
            content="Archived record",
            embedding=unit_vector(6),
            scope=MemoryScope.PROJECT,
            project_id="project-a",
            status=MemoryStatus.ARCHIVED,
        )
    )

    assert await repository.get_by_id(archived.id, "project-a") == archived
    assert await repository.get_by_id(archived.id, "project-b") is None
    assert await repository.get_active_by_logical_id(archived.logical_id, "project-a") is None
