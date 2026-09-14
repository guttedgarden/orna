"""Application service for deterministic hybrid memory retrieval."""

import asyncio
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import Settings
from app.embeddings import AsyncEmbeddingBackend
from app.models import MemoryRecord, MemorySearchResult
from app.normalizer import normalize_query_to_lexical_groups
from app.repository import MemoryRepository


class MemorySearchQuery(BaseModel):
    """Строгий application query для hybrid retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    memory_type: str | None = None
    limit: int = Field(default=5, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("memory_type")
    @classmethod
    def reject_blank_memory_type(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("memory_type must not be blank")
        if value is not None and value != value.strip():
            raise ValueError("memory_type must not have surrounding whitespace")
        return value


@dataclass(slots=True)
class _FusionCandidate:
    record: MemoryRecord
    rrf_score: float = 0.0
    rank_dense: int | None = None
    rank_lexical: int | None = None
    distance: float | None = None
    lexical_score: float | None = None


def _candidate_for(
    candidates: dict[UUID, _FusionCandidate],
    record: MemoryRecord,
) -> _FusionCandidate:
    candidate = candidates.get(record.id)
    if candidate is None:
        candidate = _FusionCandidate(record=record)
        candidates[record.id] = candidate
    elif candidate.record != record:
        raise ValueError(f"retrieval channels returned inconsistent record {record.id}")
    return candidate


def reciprocal_rank_fusion(
    dense: list[tuple[MemoryRecord, float]],
    lexical: list[tuple[MemoryRecord, float]],
    *,
    k: int,
    limit: int,
) -> list[MemorySearchResult]:
    """Объединяет rankings по позиции, используя UUID как final tie-breaker."""
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    if limit < 1:
        raise ValueError("result limit must be >= 1")

    candidates: dict[UUID, _FusionCandidate] = {}
    for rank, (record, distance) in enumerate(dense, start=1):
        candidate = _candidate_for(candidates, record)
        if candidate.rank_dense is not None:
            raise ValueError(f"dense retrieval returned duplicate record {record.id}")
        candidate.rank_dense = rank
        candidate.distance = distance
        candidate.rrf_score += 1 / (k + rank)

    for rank, (record, lexical_score) in enumerate(lexical, start=1):
        candidate = _candidate_for(candidates, record)
        if candidate.rank_lexical is not None:
            raise ValueError(f"lexical retrieval returned duplicate record {record.id}")
        candidate.rank_lexical = rank
        candidate.lexical_score = lexical_score
        candidate.rrf_score += 1 / (k + rank)

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.rrf_score,
            min(
                candidate.rank_dense or float("inf"),
                candidate.rank_lexical or float("inf"),
            ),
            candidate.record.logical_id.int,
            candidate.record.id.int,
        ),
    )
    return [
        MemorySearchResult(
            id=candidate.record.id,
            logical_id=candidate.record.logical_id,
            revision=candidate.record.revision,
            scope=candidate.record.scope,
            project_id=candidate.record.project_id,
            memory_type=candidate.record.memory_type,
            status=candidate.record.status,
            content=candidate.record.content,
            tags=list(candidate.record.tags),
            identifiers=list(candidate.record.identifiers),
            provenance=dict(candidate.record.provenance),
            rrf_score=candidate.rrf_score,
            rank_dense=candidate.rank_dense,
            rank_lexical=candidate.rank_lexical,
            distance=candidate.distance,
            lexical_score=candidate.lexical_score,
        )
        for candidate in ranked[:limit]
    ]


class MemorySearchService:
    """Запускает dense/lexical branches параллельно и применяет RRF."""

    def __init__(
        self,
        repository: MemoryRepository,
        embeddings: AsyncEmbeddingBackend,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._settings = settings

    async def search(
        self,
        search_query: MemorySearchQuery,
        project_id: str | None,
    ) -> list[MemorySearchResult]:
        lexical_query_groups = normalize_query_to_lexical_groups(search_query.query)
        query_embedding = await self._embeddings.embed_query(search_query.query)
        candidate_limit = max(
            self._settings.retrieval_candidate_pool_size,
            search_query.limit,
        )

        # Repository methods сами забирают разные connections из pool.
        dense, lexical = await asyncio.gather(
            self._repository.search_dense(
                query_embedding,
                project_id,
                candidate_limit,
                memory_type=search_query.memory_type,
            ),
            self._repository.search_lexical(
                lexical_query_groups,
                project_id,
                candidate_limit,
                memory_type=search_query.memory_type,
            ),
        )
        return reciprocal_rank_fusion(
            dense,
            lexical,
            k=self._settings.rrf_k,
            limit=search_query.limit,
        )
