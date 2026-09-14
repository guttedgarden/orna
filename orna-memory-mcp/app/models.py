"""Доменные модели записей памяти и результатов поиска."""

import math
from datetime import datetime
from enum import StrEnum
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE

EMBEDDING_DIMENSION = ACTIVE_EMBEDDING_PROFILE.dimension


class MemoryScope(StrEnum):
    """Область видимости записи памяти."""

    GLOBAL = "global"
    PROJECT = "project"


class MemoryStatus(StrEnum):
    """Состояние ревизии памяти в lifecycle."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class _DomainModel(BaseModel):
    """Общие настройки строгих domain DTO."""

    # frozen обеспечивает только faux immutability: вложенные list/dict остаются mutable.
    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class _MemoryPersistenceFields(_DomainModel):
    """Общие persisted fields для чтения и вставки записи памяти."""

    id: UUID
    logical_id: UUID
    revision: int = Field(ge=1)
    supersedes_id: UUID | None
    scope: MemoryScope
    project_id: str | None
    memory_type: str
    status: MemoryStatus
    content: str
    content_hash: bytes = Field(min_length=32, max_length=32)
    tags: list[str]
    identifiers: list[str]
    lexical_source: str
    lexical_profile_version: str
    embedding: list[float] = Field(
        min_length=EMBEDDING_DIMENSION,
        max_length=EMBEDDING_DIMENSION,
    )
    embedding_model: str
    embedding_profile_version: str
    provenance: dict[str, Any]

    @field_validator("embedding")
    @classmethod
    def validate_embedding_is_finite(cls, embedding: list[float]) -> list[float]:
        """pgvector принимает только векторы с finite-компонентами."""
        if not all(math.isfinite(value) for value in embedding):
            raise ValueError("embedding values must be finite")
        if not any(value != 0.0 for value in embedding):
            raise ValueError("embedding must have a non-zero norm")
        return embedding

    @model_validator(mode="after")
    def validate_memory_invariants(self) -> Self:
        """Проверяет локальные инварианты одной ревизии памяти."""
        if self.scope is MemoryScope.GLOBAL and self.project_id is not None:
            raise ValueError("global memory cannot have a project_id")
        if self.scope is MemoryScope.PROJECT and (
            self.project_id is None or not self.project_id.strip()
        ):
            raise ValueError("project memory requires a non-empty project_id")

        if self.revision == 1 and self.supersedes_id is not None:
            raise ValueError("revision 1 cannot supersede another memory")
        if self.revision > 1 and self.supersedes_id is None:
            raise ValueError("revision greater than 1 requires supersedes_id")
        if self.supersedes_id == self.id:
            raise ValueError("memory cannot supersede itself")

        return self


class MemoryRecord(_MemoryPersistenceFields):
    """Полная строка ``memories``, гидратированная из PostgreSQL."""

    created_at: datetime
    status_changed_at: datetime | None


class MemoryInsertRecord(_MemoryPersistenceFields):
    """Полностью подготовленная запись для передачи в repository."""


class MemorySearchResult(_DomainModel):
    """Объединённый результат dense и lexical retrieval."""

    id: UUID
    logical_id: UUID
    revision: int = Field(ge=1)
    scope: MemoryScope
    project_id: str | None
    memory_type: str
    status: MemoryStatus
    content: str
    tags: list[str]
    identifiers: list[str]
    provenance: dict[str, Any]
    rrf_score: float
    rank_dense: int | None = Field(ge=1)
    rank_lexical: int | None = Field(ge=1)
    distance: float | None
    lexical_score: float | None

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        """Не допускает результата поиска с некорректной областью видимости."""
        if self.scope is MemoryScope.GLOBAL and self.project_id is not None:
            raise ValueError("global memory cannot have a project_id")
        if self.scope is MemoryScope.PROJECT and (
            self.project_id is None or not self.project_id.strip()
        ):
            raise ValueError("project memory requires a non-empty project_id")
        return self
