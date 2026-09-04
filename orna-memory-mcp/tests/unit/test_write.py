from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models import EMBEDDING_DIMENSION, MemoryScope, MemoryStatus
from app.normalizer import build_lexical_source, canonical_content_hash
from app.write import MemoryAddCommand, MemoryWriteService, ProjectContextError


def embedding() -> list[float]:
    return [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)


def command(**overrides: object) -> MemoryAddCommand:
    values: dict[str, object] = {
        "content": "Use PostgreSQL for migration tests.",
        "scope": MemoryScope.PROJECT,
        "memory_type": "decision",
        "tags": ["database"],
        "identifiers": ["MigrationRunner"],
        "provenance": {"source": "unit-test"},
    }
    values.update(overrides)
    return MemoryAddCommand.model_validate(values)


async def test_add_prepares_complete_record_before_repository_insert():
    events: list[str] = []
    embeddings = MagicMock()

    async def embed_memory(content: str) -> list[float]:
        events.append(f"embed:{content}")
        return embedding()

    repository = MagicMock()

    async def insert(record):
        events.append("insert")
        return record

    embeddings.embed_memory = AsyncMock(side_effect=embed_memory)
    repository.insert = AsyncMock(side_effect=insert)
    settings = Settings(_env_file=None)
    service = MemoryWriteService(repository, embeddings, settings)
    add_command = command()

    result = await service.add(add_command, project_id="project-a")

    assert events == [f"embed:{add_command.content}", "insert"]
    assert result.id != result.logical_id
    assert result.revision == 1
    assert result.supersedes_id is None
    assert result.status is MemoryStatus.ACTIVE
    assert result.scope is MemoryScope.PROJECT
    assert result.project_id == "project-a"
    assert result.content_hash == canonical_content_hash(add_command.content)
    assert result.lexical_source == build_lexical_source(
        add_command.content,
        add_command.tags,
        add_command.identifiers,
    )
    assert result.embedding == embedding()
    assert result.embedding_model == settings.embedding_model
    assert result.embedding_profile_version == settings.embedding_profile_version
    assert result.lexical_profile_version == settings.lexical_profile_version


async def test_global_add_does_not_persist_project_context():
    embeddings = MagicMock()
    embeddings.embed_memory = AsyncMock(return_value=embedding())
    repository = MagicMock()
    repository.insert = AsyncMock(side_effect=lambda record: record)
    service = MemoryWriteService(repository, embeddings, Settings(_env_file=None))

    result = await service.add(command(scope=MemoryScope.GLOBAL), project_id="project-a")

    assert result.scope is MemoryScope.GLOBAL
    assert result.project_id is None


@pytest.mark.parametrize("project_id", [None, "", "   ", " project-a"])
async def test_project_add_requires_canonical_project_context(project_id):
    embeddings = MagicMock()
    embeddings.embed_memory = AsyncMock(return_value=embedding())
    repository = MagicMock()
    repository.insert = AsyncMock()
    service = MemoryWriteService(repository, embeddings, Settings(_env_file=None))

    with pytest.raises(ProjectContextError, match="project context"):
        await service.add(command(), project_id=project_id)

    embeddings.embed_memory.assert_not_awaited()
    repository.insert.assert_not_awaited()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"content": "   "}, "content"),
        ({"memory_type": ""}, "memory_type"),
        ({"memory_type": " decision"}, "memory_type"),
        ({"tags": ["valid", " "]}, "tags"),
        ({"identifiers": [""]}, "identifiers"),
        ({"provenance": {"invalid": object()}}, "(?i)json"),
        ({"unknown": True}, "extra_forbidden"),
    ],
)
def test_add_command_rejects_invalid_input(overrides, message):
    with pytest.raises(ValidationError, match=message):
        command(**overrides)
