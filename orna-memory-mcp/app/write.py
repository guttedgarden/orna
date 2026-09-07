"""Application service for creating the first revision of a memory."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from app.config import Settings
from app.embeddings import AsyncEmbeddingBackend
from app.identifiers import new_memory_id
from app.models import MemoryInsertRecord, MemoryRecord, MemoryScope, MemoryStatus
from app.normalizer import build_lexical_source, canonical_content_hash
from app.repository import MemoryRepository


class ProjectContextError(ValueError):
    """Project-scoped write не имеет валидного server-provided context."""


class MemoryAddCommand(BaseModel):
    """Строгий application command без server-owned project context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    scope: MemoryScope
    memory_type: str
    tags: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("memory_type")
    @classmethod
    def require_canonical_memory_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        if value != value.strip():
            raise ValueError("memory_type must not have surrounding whitespace")
        return value

    @field_validator("tags", "identifiers")
    @classmethod
    def reject_blank_list_items(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("items must not be blank")
        return values


class MemoryWriteService:
    """Готовит immutable revision и передаёт persistence repository."""

    def __init__(
        self,
        repository: MemoryRepository,
        embeddings: AsyncEmbeddingBackend,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._settings = settings

    async def add(self, command: MemoryAddCommand, project_id: str | None) -> MemoryRecord:
        """Создаёт первую active revision; project_id приходит только из server context."""
        record_project_id: str | None = None
        if command.scope is MemoryScope.PROJECT:
            if project_id is None or not project_id.strip():
                raise ProjectContextError("project-scoped memory requires project context")
            if project_id != project_id.strip():
                raise ProjectContextError("project context must not have surrounding whitespace")
            record_project_id = project_id

        content_hash = canonical_content_hash(command.content)
        lexical_source = build_lexical_source(
            command.content,
            command.tags,
            command.identifiers,
        )

        # ONNX inference завершается до того, как repository заберёт DB connection.
        embedding = await self._embeddings.embed_memory(command.content)
        record = MemoryInsertRecord(
            id=new_memory_id(),
            logical_id=new_memory_id(),
            revision=1,
            supersedes_id=None,
            scope=command.scope,
            project_id=record_project_id,
            memory_type=command.memory_type,
            status=MemoryStatus.ACTIVE,
            content=command.content,
            content_hash=content_hash,
            tags=list(command.tags),
            identifiers=list(command.identifiers),
            lexical_source=lexical_source,
            lexical_profile_version=self._settings.lexical_profile_version,
            embedding=embedding,
            embedding_model=self._settings.embedding_model,
            embedding_profile_version=self._settings.embedding_profile_version,
            provenance=dict(command.provenance),
        )
        return await self._repository.insert(record)
