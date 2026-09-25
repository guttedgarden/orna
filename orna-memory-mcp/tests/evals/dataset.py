"""Строгая загрузка и валидация versioned retrieval datasets."""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class DatasetValidationError(ValueError):
    """Dataset нарушает schema или cross-file invariant."""


class _DatasetModel(BaseModel):
    """Общие strict-настройки eval DTO."""

    model_config = ConfigDict(extra="forbid", frozen=True)


NonEmptyText = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

EXPECTED_FILES = frozenset({"corpus.jsonl", "memory_dev.jsonl", "memory_holdout.jsonl"})
EXPECTED_SPLITS = frozenset({"dev", "holdout"})
ALLOWED_SLICES = frozenset(
    {
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
)


class CorpusRecord(_DatasetModel):
    """Одна обезличенная memory с устойчивым fixture key."""

    memory_key: NonEmptyText
    id: UUID
    logical_id: UUID
    revision: int = Field(ge=1)
    supersedes_id: UUID | None
    status: Literal["active", "superseded", "archived"]
    scope: Literal["global", "project"]
    project_id: str | None
    memory_type: NonEmptyText
    content: NonEmptyText
    tags: list[NonEmptyText]
    identifiers: list[NonEmptyText]
    source_ref: NonEmptyText

    @field_validator("id", "logical_id", "supersedes_id")
    @classmethod
    def validate_uuidv7(cls, value: UUID | None) -> UUID | None:
        """Eval fixtures используют фиксированные UUIDv7 для стабильного tie-break."""

        if value is not None and value.version != 7:
            raise ValueError("fixture identifiers must be UUIDv7")
        return value


class QueryRecord(_DatasetModel):
    """Один retrieval case и его независимые relevance labels."""

    case_id: NonEmptyText
    split_group: NonEmptyText
    query: NonEmptyText
    project_id: NonEmptyText
    memory_type: str | None
    query_language: Literal["ru", "en", "mixed"]
    target_language: Literal["ru", "en", "mixed"]
    slices: list[NonEmptyText] = Field(min_length=1)
    relevance: dict[NonEmptyText, Literal[1, 2]]
    forbidden: list[NonEmptyText]


class SourceRecord(_DatasetModel):
    """Обезличенное происхождение группы corpus records."""

    source_ref: NonEmptyText
    description: NonEmptyText


class FileRecord(_DatasetModel):
    """Hash и число JSONL records."""

    records: int = Field(ge=1)
    sha256: Sha256Hex


class SplitRecord(_DatasetModel):
    """Frozen агрегаты query split."""

    records: int = Field(ge=1)
    positive: int = Field(ge=0)
    negative: int = Field(ge=0)
    slices: dict[NonEmptyText, int]


class DatasetManifest(_DatasetModel):
    """Версия, provenance, hashes и split summary набора."""

    schema_version: Literal[1]
    dataset_version: NonEmptyText
    frozen_at: date
    split_policy: NonEmptyText
    holdout_policy: NonEmptyText
    sources: list[SourceRecord] = Field(min_length=1)
    files: dict[str, FileRecord]
    splits: dict[str, SplitRecord]


class RetrievalDataset(_DatasetModel):
    """Полностью загруженный retrieval dataset."""

    manifest: DatasetManifest
    corpus: tuple[CorpusRecord, ...]
    dev: tuple[QueryRecord, ...]
    holdout: tuple[QueryRecord, ...]


class RetrievalSplit(_DatasetModel):
    """Corpus и labels только одного явно выбранного split."""

    manifest: DatasetManifest
    corpus: tuple[CorpusRecord, ...]
    name: Literal["dev", "holdout"]
    queries: tuple[QueryRecord, ...]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"cannot read {path.name}: {exc}") from exc


def _read_jsonl(path: Path) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"cannot read {path.name}: {exc}") from exc


def _verify_file(path: Path, expected: FileRecord, actual_records: int) -> None:
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected.sha256:
        raise DatasetValidationError(f"hash mismatch for {path.name}")
    if actual_records != expected.records:
        raise DatasetValidationError(f"record count mismatch for {path.name}")


def _ensure_unique(values: list[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise DatasetValidationError(f"duplicate {label}")


def _validate_manifest_shape(manifest: DatasetManifest) -> None:
    if set(manifest.files) != EXPECTED_FILES:
        raise DatasetValidationError("manifest files must match the three dataset JSONL files")
    if set(manifest.splits) != EXPECTED_SPLITS:
        raise DatasetValidationError("manifest splits must contain dev and holdout")
    _ensure_unique([source.source_ref for source in manifest.sources], "source_ref")


def _validate_corpus(
    corpus: tuple[CorpusRecord, ...],
    manifest: DatasetManifest,
) -> dict[str, CorpusRecord]:
    _ensure_unique([record.memory_key for record in corpus], "memory_key")
    _ensure_unique([record.id for record in corpus], "physical id")
    _ensure_unique(
        [(record.logical_id, record.revision) for record in corpus],
        "logical revision",
    )
    supersedes_ids = [record.supersedes_id for record in corpus if record.supersedes_id]
    _ensure_unique(supersedes_ids, "supersedes_id")

    source_refs = {source.source_ref for source in manifest.sources}
    by_id = {record.id: record for record in corpus}
    active_by_logical_id: dict[UUID, int] = {}

    for record in corpus:
        if record.source_ref not in source_refs:
            raise DatasetValidationError(
                f"memory {record.memory_key} has unknown source_ref {record.source_ref}"
            )
        if record.scope == "global" and record.project_id is not None:
            raise DatasetValidationError("global memory cannot have a project_id")
        if record.scope == "project" and (
            record.project_id is None or not record.project_id.strip()
        ):
            raise DatasetValidationError("project memory requires a non-empty project_id")
        if record.revision == 1 and record.supersedes_id is not None:
            raise DatasetValidationError("revision 1 cannot have supersedes_id")
        if record.revision > 1:
            if record.supersedes_id is None:
                raise DatasetValidationError("revision greater than 1 requires supersedes_id")
            predecessor = by_id.get(record.supersedes_id)
            if predecessor is None:
                raise DatasetValidationError(
                    f"memory {record.memory_key} has unknown supersedes_id"
                )
            if (
                predecessor.logical_id != record.logical_id
                or predecessor.revision != record.revision - 1
            ):
                raise DatasetValidationError(
                    f"memory {record.memory_key} must supersede the immediately previous revision"
                )
            if predecessor.status != "superseded":
                raise DatasetValidationError("a superseded predecessor must have superseded status")
        if record.status == "active":
            active_by_logical_id[record.logical_id] = (
                active_by_logical_id.get(record.logical_id, 0) + 1
            )

    if any(count > 1 for count in active_by_logical_id.values()):
        raise DatasetValidationError("multiple active revisions for one logical_id")
    return {record.memory_key: record for record in corpus}


def _validate_queries(
    dev: tuple[QueryRecord, ...],
    holdout: tuple[QueryRecord, ...],
    corpus_by_key: dict[str, CorpusRecord],
) -> None:
    queries = dev + holdout
    _ensure_unique([query.case_id for query in queries], "case_id")

    leaked_groups = {query.split_group for query in dev} & {query.split_group for query in holdout}
    if leaked_groups:
        leaked = ", ".join(sorted(leaked_groups))
        raise DatasetValidationError(f"split_group leakage: {leaked}")

    dev_relevance = {key for query in dev for key in query.relevance}
    holdout_relevance = {key for query in holdout for key in query.relevance}
    leaked_relevance = dev_relevance & holdout_relevance
    if leaked_relevance:
        leaked = ", ".join(sorted(leaked_relevance))
        raise DatasetValidationError(f"relevance leakage between dev and holdout: {leaked}")

    for query in queries:
        if not query.project_id.strip():
            raise DatasetValidationError(f"query {query.case_id} requires a non-empty project_id")
        if query.memory_type is not None:
            if not query.memory_type.strip():
                raise DatasetValidationError("memory_type must not be blank")
            if query.memory_type != query.memory_type.strip():
                raise DatasetValidationError("memory_type must not have surrounding whitespace")
        if len(query.slices) != len(set(query.slices)):
            raise DatasetValidationError(f"query {query.case_id} has a duplicate slice")
        if len(query.forbidden) != len(set(query.forbidden)):
            raise DatasetValidationError(f"query {query.case_id} has a duplicate forbidden key")
        unknown_slices = set(query.slices) - ALLOWED_SLICES
        if unknown_slices:
            raise DatasetValidationError(
                f"query {query.case_id} has unknown slice: {', '.join(sorted(unknown_slices))}"
            )
        is_negative = not query.relevance
        has_negative_slice = "negative" in query.slices
        if is_negative and not has_negative_slice:
            raise DatasetValidationError(f"negative query {query.case_id} requires negative slice")
        if not is_negative and has_negative_slice:
            raise DatasetValidationError(
                f"positive query {query.case_id} cannot use negative slice"
            )
        relevant_keys = set(query.relevance)
        forbidden_keys = set(query.forbidden)
        overlap = relevant_keys & forbidden_keys
        if overlap:
            raise DatasetValidationError(
                f"query {query.case_id} marks a key both relevant and forbidden"
            )
        for key, reference_kind in (
            *((key, "relevance") for key in relevant_keys),
            *((key, "forbidden") for key in forbidden_keys),
        ):
            if key not in corpus_by_key:
                raise DatasetValidationError(
                    f"query {query.case_id} has unknown {reference_kind} key {key}"
                )
        for key in relevant_keys:
            relevant = corpus_by_key[key]
            visible = relevant.scope == "global" or relevant.project_id == query.project_id
            type_matches = query.memory_type is None or relevant.memory_type == query.memory_type
            if relevant.status != "active" or not visible or not type_matches:
                raise DatasetValidationError(
                    f"relevance key {key} is not visible to query {query.case_id}"
                )


def _computed_split_record(queries: tuple[QueryRecord, ...]) -> dict[str, Any]:
    slices: dict[str, int] = {}
    for query in queries:
        for slice_name in query.slices:
            slices[slice_name] = slices.get(slice_name, 0) + 1
    positive = sum(bool(query.relevance) for query in queries)
    return {
        "records": len(queries),
        "positive": positive,
        "negative": len(queries) - positive,
        "slices": dict(sorted(slices.items())),
    }


def _validate_split_summaries(
    manifest: DatasetManifest,
    dev: tuple[QueryRecord, ...],
    holdout: tuple[QueryRecord, ...],
) -> None:
    for split_name, queries in (("dev", dev), ("holdout", holdout)):
        expected = manifest.splits[split_name].model_dump()
        if expected != _computed_split_record(queries):
            raise DatasetValidationError(f"split summary mismatch for {split_name}")


def load_retrieval_dataset(root: Path) -> RetrievalDataset:
    """Загружает dataset и проверяет manifest-bound file integrity."""

    try:
        manifest = DatasetManifest.model_validate(_read_json(root / "dataset-manifest.json"))
    except ValidationError as exc:
        raise DatasetValidationError(f"invalid dataset manifest: {exc}") from exc
    _validate_manifest_shape(manifest)
    payloads = {
        "corpus.jsonl": _read_jsonl(root / "corpus.jsonl"),
        "memory_dev.jsonl": _read_jsonl(root / "memory_dev.jsonl"),
        "memory_holdout.jsonl": _read_jsonl(root / "memory_holdout.jsonl"),
    }
    for file_name, payload in payloads.items():
        expected = manifest.files.get(file_name)
        if expected is None:
            raise DatasetValidationError(f"manifest is missing {file_name}")
        _verify_file(root / file_name, expected, len(payload))

    try:
        dataset = RetrievalDataset(
            manifest=manifest,
            corpus=tuple(
                CorpusRecord.model_validate(record) for record in payloads["corpus.jsonl"]
            ),
            dev=tuple(
                QueryRecord.model_validate(record) for record in payloads["memory_dev.jsonl"]
            ),
            holdout=tuple(
                QueryRecord.model_validate(record) for record in payloads["memory_holdout.jsonl"]
            ),
        )
    except ValidationError as exc:
        raise DatasetValidationError(f"invalid dataset record: {exc}") from exc

    corpus_by_key = _validate_corpus(dataset.corpus, manifest)
    _validate_queries(dataset.dev, dataset.holdout, corpus_by_key)
    _validate_split_summaries(manifest, dataset.dev, dataset.holdout)
    return dataset


def load_retrieval_split(
    root: Path,
    split: Literal["dev", "holdout"] = "dev",
) -> RetrievalSplit:
    """Загружает только выбранные labels, не открывая файл другого split."""

    try:
        manifest = DatasetManifest.model_validate(_read_json(root / "dataset-manifest.json"))
    except ValidationError as exc:
        raise DatasetValidationError(f"invalid dataset manifest: {exc}") from exc
    _validate_manifest_shape(manifest)

    query_file = f"memory_{split}.jsonl"
    corpus_payload = _read_jsonl(root / "corpus.jsonl")
    query_payload = _read_jsonl(root / query_file)
    _verify_file(root / "corpus.jsonl", manifest.files["corpus.jsonl"], len(corpus_payload))
    _verify_file(root / query_file, manifest.files[query_file], len(query_payload))

    try:
        selected = RetrievalSplit(
            manifest=manifest,
            corpus=tuple(CorpusRecord.model_validate(record) for record in corpus_payload),
            name=split,
            queries=tuple(QueryRecord.model_validate(record) for record in query_payload),
        )
    except ValidationError as exc:
        raise DatasetValidationError(f"invalid dataset record: {exc}") from exc

    corpus_by_key = _validate_corpus(selected.corpus, manifest)
    if split == "dev":
        _validate_queries(selected.queries, (), corpus_by_key)
    else:
        _validate_queries((), selected.queries, corpus_by_key)
    if manifest.splits[split].model_dump() != _computed_split_record(selected.queries):
        raise DatasetValidationError(f"split summary mismatch for {split}")
    return selected
