from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models import (
    EMBEDDING_DIMENSION,
    MemoryInsertRecord,
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryStatus,
)


def memory_data(**overrides: object) -> dict[str, object]:
    record_id = uuid4()
    data: dict[str, object] = {
        "id": record_id,
        "logical_id": uuid4(),
        "revision": 1,
        "supersedes_id": None,
        "scope": MemoryScope.GLOBAL,
        "project_id": None,
        "memory_type": "decision",
        "status": MemoryStatus.ACTIVE,
        "content": "PostgreSQL is the storage backend.",
        "content_hash": b"x" * 32,
        "tags": ["postgresql"],
        "identifiers": ["MemoryRepository"],
        "lexical_source": "PostgreSQL MemoryRepository memory repository",
        "lexical_profile_version": "lexical-v1",
        "embedding": [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
        "embedding_model": "intfloat/multilingual-e5-large",
        "embedding_profile_version": "e5-v1",
        "provenance": {"source": "test"},
    }
    data.update(overrides)
    return data


def test_memory_record_requires_every_database_field() -> None:
    data = memory_data(
        created_at=datetime.now(UTC),
        status_changed_at=None,
    )
    del data["status"]

    with pytest.raises(ValidationError, match="status"):
        MemoryRecord.model_validate(data)


def test_memory_insert_record_accepts_valid_project_revision() -> None:
    previous_id = uuid4()
    record = MemoryInsertRecord.model_validate(
        memory_data(
            revision=2,
            supersedes_id=previous_id,
            scope="project",
            project_id="orna",
        )
    )

    assert record.scope is MemoryScope.PROJECT
    assert record.status is MemoryStatus.ACTIVE
    assert record.supersedes_id == previous_id


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"revision": 0}, "greater than or equal to 1"),
        ({"revision": 1, "supersedes_id": uuid4()}, "revision 1 cannot supersede"),
        ({"revision": 2, "supersedes_id": None}, "revision greater than 1"),
        ({"scope": "global", "project_id": "orna"}, "global memory cannot have"),
        ({"scope": "project", "project_id": None}, "project memory requires"),
        ({"scope": "project", "project_id": "   "}, "project memory requires"),
        ({"content_hash": b"short"}, "at least 32 bytes"),
        ({"embedding": []}, "at least 1024 items"),
        ({"embedding": [0.0] * (EMBEDDING_DIMENSION - 1)}, "at least 1024 items"),
        ({"embedding": [0.0] * EMBEDDING_DIMENSION + [0.0]}, "at most 1024 items"),
        ({"embedding": [float("nan")] * EMBEDDING_DIMENSION}, "finite"),
        ({"embedding": [float("inf")] * EMBEDDING_DIMENSION}, "finite"),
        ({"embedding": [0.0] * EMBEDDING_DIMENSION}, "non-zero norm"),
    ],
)
def test_memory_insert_record_rejects_invalid_domain_state(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        MemoryInsertRecord.model_validate(memory_data(**overrides))


def test_memory_insert_record_rejects_self_supersede() -> None:
    record_id = uuid4()

    with pytest.raises(ValidationError, match="cannot supersede itself"):
        MemoryInsertRecord.model_validate(
            memory_data(id=record_id, revision=2, supersedes_id=record_id)
        )


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MemoryInsertRecord.model_validate(memory_data(unknown_field=True))


def test_memory_record_prevents_field_reassignment() -> None:
    record = MemoryRecord.model_validate(
        memory_data(created_at=datetime.now(UTC), status_changed_at=None)
    )

    with pytest.raises(ValidationError, match="frozen"):
        record.content = "Changed"


def test_memory_search_result_represents_both_retrieval_channels() -> None:
    result = MemorySearchResult(
        id=uuid4(),
        logical_id=uuid4(),
        revision=1,
        scope=MemoryScope.PROJECT,
        project_id="orna",
        memory_type="decision",
        status=MemoryStatus.ACTIVE,
        content="Use PostgreSQL.",
        tags=["database"],
        identifiers=["MemoryRepository"],
        provenance={"source": "test"},
        rrf_score=0.032,
        rank_dense=1,
        rank_lexical=2,
        distance=0.1,
        lexical_score=0.5,
    )

    assert result.rank_dense == 1
    assert result.rank_lexical == 2
