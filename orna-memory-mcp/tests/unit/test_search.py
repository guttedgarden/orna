import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models import EMBEDDING_DIMENSION, MemoryRecord, MemoryScope, MemoryStatus
from app.search import MemorySearchQuery, MemorySearchService, reciprocal_rank_fusion


def record(id_value: int, content: str, logical_id_value: int | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=UUID(int=id_value),
        logical_id=UUID(int=100 + id_value if logical_id_value is None else logical_id_value),
        revision=1,
        supersedes_id=None,
        scope=MemoryScope.GLOBAL,
        project_id=None,
        memory_type="decision",
        status=MemoryStatus.ACTIVE,
        content=content,
        content_hash=b"x" * 32,
        tags=[],
        identifiers=[],
        lexical_source=content,
        lexical_profile_version="lexical-v1",
        embedding=[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
        embedding_model="intfloat/multilingual-e5-large",
        embedding_profile_version="e5-v1",
        provenance={},
        created_at=datetime.now(UTC),
        status_changed_at=None,
    )


def test_rrf_combines_channels_and_preserves_channel_metadata():
    first = record(1, "dense only")
    shared = record(2, "shared")
    lexical_first = record(3, "lexical first")

    results = reciprocal_rank_fusion(
        dense=[(first, 0.1), (shared, 0.2), (lexical_first, 0.3)],
        lexical=[(lexical_first, 0.9), (shared, 0.8)],
        k=60,
        limit=3,
    )

    assert [result.id for result in results] == [lexical_first.id, shared.id, first.id]
    assert results[0].rank_dense == 3
    assert results[0].rank_lexical == 1
    assert results[0].distance == pytest.approx(0.3)
    assert results[0].lexical_score == pytest.approx(0.9)
    assert results[0].rrf_score == pytest.approx(1 / 63 + 1 / 61)
    assert results[2].rank_lexical is None
    assert results[2].lexical_score is None


def test_rrf_uses_logical_then_physical_id_as_final_tie_breakers():
    lower_logical_id = record(2, "lexical", logical_id_value=101)
    higher_logical_id = record(1, "dense", logical_id_value=102)

    results = reciprocal_rank_fusion(
        dense=[(higher_logical_id, 0.1)],
        lexical=[(lower_logical_id, 0.9)],
        k=60,
        limit=2,
    )

    assert [result.id for result in results] == [lower_logical_id.id, higher_logical_id.id]


@pytest.mark.parametrize(("k", "limit"), [(0, 1), (60, 0)])
def test_rrf_rejects_non_positive_parameters(k, limit):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], k=k, limit=limit)


async def test_search_normalizes_query_and_runs_retrieval_channels_concurrently():
    repository = MagicMock()
    embeddings = MagicMock()
    query_vector = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
    embeddings.embed_query = AsyncMock(return_value=query_vector)
    both_started = asyncio.Event()
    started = 0

    async def branch_result(*_args, **_kwargs):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return []

    repository.search_dense = AsyncMock(side_effect=branch_result)
    repository.search_lexical = AsyncMock(side_effect=branch_result)
    settings = Settings(retrieval_candidate_pool_size=17, _env_file=None)
    service = MemorySearchService(repository, embeddings, settings)

    results = await service.search(
        MemorySearchQuery(query="ResponseProviderExecutor", memory_type="decision", limit=5),
        project_id="project-a",
    )

    assert results == []
    embeddings.embed_query.assert_awaited_once_with("ResponseProviderExecutor")
    repository.search_dense.assert_awaited_once_with(
        query_vector,
        "project-a",
        17,
        memory_type="decision",
    )
    repository.search_lexical.assert_awaited_once_with(
        [("responseproviderexecutor", "response provider executor")],
        "project-a",
        17,
        memory_type="decision",
    )


def test_search_query_rejects_blank_query_and_invalid_limit():
    with pytest.raises(ValidationError, match="query"):
        MemorySearchQuery(query="   ")
    with pytest.raises(ValidationError, match="limit"):
        MemorySearchQuery(query="database", limit=0)
    with pytest.raises(ValidationError, match="memory_type"):
        MemorySearchQuery(query="database", memory_type=" ")
    with pytest.raises(ValidationError, match="memory_type"):
        MemorySearchQuery(query="database", memory_type="decision ")
