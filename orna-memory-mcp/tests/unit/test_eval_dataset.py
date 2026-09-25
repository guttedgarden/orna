import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.evals.dataset import DatasetValidationError, load_retrieval_dataset

UUID_ROOT = "018f0c7a-1234-7abc-8def-000000000001"
UUID_PROJECT = "018f0c7a-1234-7abc-8def-000000000002"
UUID_OLD = "018f0c7a-1234-7abc-8def-000000000003"
UUID_CURRENT = "018f0c7a-1234-7abc-8def-000000000004"
LOGICAL_REVISION = "018f0c7a-1234-7abc-8def-000000000005"


def _memory(**overrides: object) -> dict[str, Any]:
    record: dict[str, Any] = {
        "memory_key": "storage-global",
        "id": UUID_ROOT,
        "logical_id": UUID_ROOT,
        "revision": 1,
        "supersedes_id": None,
        "status": "active",
        "scope": "global",
        "project_id": None,
        "memory_type": "decision",
        "content": "Orna stores memories in PostgreSQL with pgvector.",
        "tags": ["postgresql", "pgvector"],
        "identifiers": ["MemoryRepository"],
        "source_ref": "phase0-memory-journal",
    }
    record.update(overrides)
    return record


def _query(**overrides: object) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": "dev-storage-001",
        "split_group": "storage-choice",
        "query": "Which database stores Orna memories?",
        "project_id": "eval-a",
        "memory_type": None,
        "query_language": "en",
        "target_language": "en",
        "slices": ["semantic"],
        "relevance": {"storage-global": 2},
        "forbidden": [],
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )
    path.write_text(payload, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    slices: dict[str, int] = {}
    for record in records:
        for slice_name in record["slices"]:
            slices[slice_name] = slices.get(slice_name, 0) + 1
    positive = sum(bool(record["relevance"]) for record in records)
    return {
        "records": len(records),
        "positive": positive,
        "negative": len(records) - positive,
        "slices": dict(sorted(slices.items())),
    }


def _write_dataset(
    root: Path,
    *,
    corpus: list[dict[str, Any]] | None = None,
    dev: list[dict[str, Any]] | None = None,
    holdout: list[dict[str, Any]] | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    corpus = corpus or [_memory()]
    dev = dev or [_query()]
    holdout = holdout or [
        _query(
            case_id="holdout-storage-001",
            split_group="storage-runtime",
            query="Which Redis cluster stores Orna vectors?",
            query_language="en",
            target_language="en",
            slices=["negative"],
            relevance={},
            forbidden=["storage-global"],
        )
    ]
    root.mkdir()
    files = {
        "corpus.jsonl": corpus,
        "memory_dev.jsonl": dev,
        "memory_holdout.jsonl": holdout,
    }
    for file_name, records in files.items():
        _write_jsonl(root / file_name, records)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": "phase-2-v1-test",
        "frozen_at": "2026-09-15",
        "split_policy": "Split incident/topic/session groups before paraphrases.",
        "holdout_policy": "Never use holdout for tuning or distractor selection.",
        "sources": [
            {
                "source_ref": "phase0-memory-journal",
                "description": "Anonymized and repository-verified Phase 0 seed.",
            }
        ],
        "files": {
            file_name: {"records": len(records), "sha256": _sha256(root / file_name)}
            for file_name, records in files.items()
        },
        "splits": {"dev": _split_stats(dev), "holdout": _split_stats(holdout)},
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    (root / "dataset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def test_load_retrieval_dataset_returns_validated_records(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "retrieval")

    dataset = load_retrieval_dataset(root)

    assert dataset.manifest.dataset_version == "phase-2-v1-test"
    assert [record.memory_key for record in dataset.corpus] == ["storage-global"]
    assert [case.case_id for case in dataset.dev] == ["dev-storage-001"]
    assert [case.case_id for case in dataset.holdout] == ["holdout-storage-001"]


def test_dataset_rejects_non_uuidv7_identifiers(tmp_path: Path) -> None:
    root = _write_dataset(
        tmp_path / "retrieval",
        corpus=[_memory(id="00000000-0000-4000-8000-000000000001")],
    )

    with pytest.raises(DatasetValidationError, match="UUIDv7"):
        load_retrieval_dataset(root)


@pytest.mark.parametrize(
    ("corpus", "message"),
    [
        ([_memory(), _memory(id=UUID_PROJECT)], "duplicate memory_key"),
        (
            [_memory(), _memory(memory_key="project-copy", id=UUID_ROOT)],
            "duplicate physical id",
        ),
        (
            [_memory(scope="project", project_id=None)],
            "project memory requires a non-empty project_id",
        ),
        (
            [_memory(scope="global", project_id="eval-a")],
            "global memory cannot have a project_id",
        ),
    ],
)
def test_dataset_rejects_invalid_corpus_identity_or_scope(
    tmp_path: Path,
    corpus: list[dict[str, Any]],
    message: str,
) -> None:
    root = _write_dataset(tmp_path / "retrieval", corpus=corpus)

    with pytest.raises(DatasetValidationError, match=message):
        load_retrieval_dataset(root)


@pytest.mark.parametrize(
    ("corpus", "message"),
    [
        (
            [_memory(revision=2, supersedes_id=UUID_OLD)],
            "unknown supersedes_id",
        ),
        (
            [
                _memory(
                    memory_key="old",
                    id=UUID_OLD,
                    logical_id=LOGICAL_REVISION,
                    status="superseded",
                ),
                _memory(
                    memory_key="current",
                    id=UUID_CURRENT,
                    logical_id=LOGICAL_REVISION,
                    revision=3,
                    supersedes_id=UUID_OLD,
                ),
            ],
            "immediately previous revision",
        ),
        (
            [
                _memory(memory_key="one", id=UUID_OLD, logical_id=LOGICAL_REVISION),
                _memory(memory_key="two", id=UUID_CURRENT, logical_id=LOGICAL_REVISION),
            ],
            "duplicate logical revision",
        ),
        (
            [
                _memory(
                    memory_key="old",
                    id=UUID_OLD,
                    logical_id=LOGICAL_REVISION,
                    status="superseded",
                ),
                _memory(
                    memory_key="current-a",
                    id=UUID_CURRENT,
                    logical_id=LOGICAL_REVISION,
                    revision=2,
                    supersedes_id=UUID_OLD,
                ),
                _memory(
                    memory_key="current-b",
                    id=UUID_PROJECT,
                    logical_id=LOGICAL_REVISION,
                    revision=2,
                    supersedes_id=UUID_OLD,
                ),
            ],
            "duplicate logical revision",
        ),
    ],
)
def test_dataset_rejects_broken_revision_chains(
    tmp_path: Path,
    corpus: list[dict[str, Any]],
    message: str,
) -> None:
    root = _write_dataset(tmp_path / "retrieval", corpus=corpus)

    with pytest.raises(DatasetValidationError, match=message):
        load_retrieval_dataset(root)


@pytest.mark.parametrize(
    ("dev", "message"),
    [
        ([_query(relevance={"missing": 2})], "unknown relevance key"),
        ([_query(forbidden=["missing"])], "unknown forbidden key"),
        (
            [_query(relevance={"storage-global": 2}, forbidden=["storage-global"])],
            "both relevant and forbidden",
        ),
        ([_query(slices=["invented_slice"])], "unknown slice"),
        ([_query(slices=["semantic", "semantic"])], "duplicate slice"),
        (
            [
                _query(
                    relevance={},
                    forbidden=["storage-global", "storage-global"],
                    slices=["negative"],
                )
            ],
            "duplicate forbidden",
        ),
        ([_query(project_id=" ")], "project_id"),
        ([_query(memory_type=" ")], "memory_type must not be blank"),
        ([_query(relevance={}, slices=["semantic"])], "negative slice"),
        ([_query(slices=["semantic", "negative"])], "positive query"),
    ],
)
def test_dataset_rejects_invalid_query_references(
    tmp_path: Path,
    dev: list[dict[str, Any]],
    message: str,
) -> None:
    root = _write_dataset(tmp_path / "retrieval", dev=dev)

    with pytest.raises(DatasetValidationError, match=message):
        load_retrieval_dataset(root)


def test_dataset_rejects_relevance_that_search_cannot_return(tmp_path: Path) -> None:
    corpus = [
        _memory(
            memory_key="other-project",
            id=UUID_PROJECT,
            scope="project",
            project_id="eval-b",
        )
    ]
    root = _write_dataset(
        tmp_path / "retrieval",
        corpus=corpus,
        dev=[_query(relevance={"other-project": 2})],
        holdout=[
            _query(
                case_id="holdout-negative-001",
                split_group="negative-runtime",
                relevance={},
                forbidden=["other-project"],
                slices=["negative", "visibility"],
            )
        ],
    )

    with pytest.raises(DatasetValidationError, match="not visible to query"):
        load_retrieval_dataset(root)


def test_dataset_rejects_split_group_leakage(tmp_path: Path) -> None:
    root = _write_dataset(
        tmp_path / "retrieval",
        holdout=[_query(case_id="holdout-storage-001", split_group="storage-choice")],
    )

    with pytest.raises(DatasetValidationError, match="split_group leakage"):
        load_retrieval_dataset(root)


def test_dataset_rejects_relevant_memory_leakage_between_splits(tmp_path: Path) -> None:
    root = _write_dataset(
        tmp_path / "retrieval",
        holdout=[
            _query(
                case_id="holdout-storage-001",
                split_group="storage-runtime",
            )
        ],
    )

    with pytest.raises(DatasetValidationError, match="relevance leakage"):
        load_retrieval_dataset(root)


def test_dataset_rejects_duplicate_case_ids_across_splits(tmp_path: Path) -> None:
    root = _write_dataset(
        tmp_path / "retrieval",
        holdout=[_query(split_group="storage-runtime")],
    )

    with pytest.raises(DatasetValidationError, match="duplicate case_id"):
        load_retrieval_dataset(root)


def test_dataset_rejects_unknown_source_ref(tmp_path: Path) -> None:
    root = _write_dataset(
        tmp_path / "retrieval",
        corpus=[_memory(source_ref="unregistered-source")],
    )

    with pytest.raises(DatasetValidationError, match="unknown source_ref"):
        load_retrieval_dataset(root)


def test_dataset_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "retrieval")
    with (root / "memory_dev.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(DatasetValidationError, match=r"hash mismatch for memory_dev\.jsonl"):
        load_retrieval_dataset(root)


def test_dataset_rejects_manifest_split_summary_mismatch(tmp_path: Path) -> None:
    def mutate_manifest(manifest: dict[str, Any]) -> None:
        manifest["splits"]["dev"]["positive"] = 0

    root = _write_dataset(tmp_path / "retrieval", mutate_manifest=mutate_manifest)

    with pytest.raises(DatasetValidationError, match="split summary mismatch for dev"):
        load_retrieval_dataset(root)


def test_tracked_dataset_is_frozen_balanced_and_covers_required_slices() -> None:
    root = Path(__file__).parents[1] / "retrieval"

    dataset = load_retrieval_dataset(root)

    assert len(dataset.corpus) == 36
    assert len(dataset.dev) == 24
    assert len(dataset.holdout) == 20
    assert sum(not case.relevance for case in dataset.dev) == 6
    assert sum(not case.relevance for case in dataset.holdout) == 5

    required_slices = {
        "contradiction",
        "cross_lingual",
        "en_to_ru",
        "exact_identifier",
        "lifecycle",
        "mixed",
        "negative",
        "partial_identifier",
        "path",
        "ru_to_en",
        "semantic",
        "split_identifier",
        "typo",
        "visibility",
    }
    actual_slices = {
        slice_name for case in dataset.dev + dataset.holdout for slice_name in case.slices
    }
    assert actual_slices == required_slices
    assert {case.query_language for case in dataset.dev} == {"ru", "en", "mixed"}
    assert {case.query_language for case in dataset.holdout} == {"ru", "en", "mixed"}
    assert {case.target_language for case in dataset.dev + dataset.holdout} == {"ru", "en"}
    assert any(set(case.relevance.values()) == {1, 2} for case in dataset.dev + dataset.holdout)
