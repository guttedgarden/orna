"""Только frozen selection dev: строгая загрузка без доступа к закрытым split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE
from app.embeddings import ensure_prefix, prepare_memory_text
from tests.evals.dataset import (
    CorpusRecord,
    DatasetManifest,
    DatasetValidationError,
    FileRecord,
    QueryRecord,
    SourceRecord,
    SplitRecord,
    _computed_split_record,
    _ensure_unique,
    _validate_corpus,
)

_FILES = frozenset({"corpus.jsonl", "dev.jsonl"})
_SLICES = frozenset(
    {
        "cross_lingual",
        "en_to_ru",
        "exact_identifier",
        "lifecycle",
        "memory_type",
        "multi_record",
        "near_topic",
        "negative",
        "ood",
        "partial",
        "path",
        "ru_to_en",
        "semantic",
        "typo",
        "visibility",
    }
)


class SelectionManifestBase(BaseModel):
    """Только метаданные dev; validation отсутствует даже в schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset_version: str = Field(min_length=1)
    frozen_at: date
    provenance: Literal["synthetic", "real", "mixed"]
    sources: list[SourceRecord] = Field(min_length=1)
    label_policy: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    files: dict[str, FileRecord]
    case_rationales: dict[str, str]


class SelectionManifest(SelectionManifestBase):
    dev: SplitRecord


class SelectionValidationManifest(SelectionManifestBase):
    validation: SplitRecord


@dataclass(frozen=True, slots=True)
class SelectionDev:
    manifest: SelectionManifest | SelectionValidationManifest
    corpus: tuple[CorpusRecord, ...]
    queries: tuple[QueryRecord, ...]


def _allowed_path(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise DatasetValidationError(f"selection file {name} must be inside the dev directory")
    return path


def _read_jsonl(path: Path, expected: FileRecord) -> list[dict]:
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected.sha256:
            raise DatasetValidationError(f"hash mismatch for {path.name}")
        lines = payload.decode("utf-8").splitlines()
        if len(lines) != expected.records:
            raise DatasetValidationError(f"record count mismatch for {path.name}")
        result = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                raise DatasetValidationError(f"blank line in {path.name}:{number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DatasetValidationError(f"record in {path.name}:{number} must be an object")
            result.append(value)
        return result
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"cannot read {path.name}: {exc}") from exc


def _eligible(record: CorpusRecord, query: QueryRecord) -> bool:
    return (
        record.status == "active"
        and (record.scope == "global" or record.project_id == query.project_id)
        and (query.memory_type is None or record.memory_type == query.memory_type)
    )


def _validate_query(query: QueryRecord, corpus: dict[str, CorpusRecord]) -> None:
    if not query.project_id.strip():
        raise DatasetValidationError(f"query {query.case_id} requires project_id")
    if query.memory_type is not None and (
        not query.memory_type.strip() or query.memory_type != query.memory_type.strip()
    ):
        raise DatasetValidationError(f"query {query.case_id} has invalid memory_type")
    _ensure_unique(query.slices, f"slice in {query.case_id}")
    _ensure_unique(query.forbidden, f"forbidden key in {query.case_id}")
    if unknown := set(query.slices) - _SLICES:
        raise DatasetValidationError(f"query {query.case_id} has unknown slices {sorted(unknown)}")
    negative = not query.relevance
    if negative != ("negative" in query.slices):
        label = (
            "negative query requires negative slice"
            if negative
            else "positive query cannot use negative slice"
        )
        raise DatasetValidationError(f"{label}: {query.case_id}")
    if negative != (bool({"near_topic", "ood"} & set(query.slices))):
        raise DatasetValidationError(f"query {query.case_id} has inconsistent negative kind")
    if "near_topic" in query.slices and "ood" in query.slices:
        raise DatasetValidationError(f"query {query.case_id} has two negative kinds")
    if ("cross_lingual" in query.slices) != (query.query_language != query.target_language):
        raise DatasetValidationError(f"query {query.case_id} has inconsistent language labels")
    if ("ru_to_en" in query.slices) != (
        query.query_language == "ru" and query.target_language == "en"
    ) or ("en_to_ru" in query.slices) != (
        query.query_language == "en" and query.target_language == "ru"
    ):
        raise DatasetValidationError(f"query {query.case_id} has inconsistent language direction")
    if ("partial" in query.slices) != (1 in query.relevance.values()):
        raise DatasetValidationError(f"query {query.case_id} has inconsistent partial label")
    if ("multi_record" in query.slices) != (len(query.relevance) > 1):
        raise DatasetValidationError(f"query {query.case_id} has inconsistent multi-record label")
    if ("memory_type" in query.slices) != (query.memory_type is not None):
        raise DatasetValidationError(f"query {query.case_id} has inconsistent memory_type label")
    if set(query.relevance) & set(query.forbidden):
        raise DatasetValidationError(f"query {query.case_id} overlaps relevance and forbidden")
    for key in query.relevance:
        if key not in corpus:
            raise DatasetValidationError(f"query {query.case_id} has unknown relevance key {key}")
        if not _eligible(corpus[key], query):
            raise DatasetValidationError(
                f"relevance key {key} is not visible to query {query.case_id}"
            )
    for key in query.forbidden:
        if key not in corpus:
            raise DatasetValidationError(f"query {query.case_id} has unknown forbidden key {key}")
        if _eligible(corpus[key], query):
            raise DatasetValidationError(
                f"forbidden key {key} is eligible for query {query.case_id}"
            )
    forbidden = [corpus[key] for key in query.forbidden]
    if "lifecycle" in query.slices and not any(record.status != "active" for record in forbidden):
        raise DatasetValidationError(f"query {query.case_id} has no lifecycle control")
    if "visibility" in query.slices and not any(
        record.scope == "project" and record.project_id != query.project_id for record in forbidden
    ):
        raise DatasetValidationError(f"query {query.case_id} has no visibility control")
    if "memory_type" in query.slices and not any(
        record.memory_type != query.memory_type for record in forbidden
    ):
        raise DatasetValidationError(f"query {query.case_id} has no type control")


def validate_e5_inputs(dataset: SelectionDev, tokenizer_path: Path) -> dict[str, int]:
    """Локальный tokenizer-only precondition; модели и сеть не используются."""

    from tokenizers import Tokenizer

    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:  # tokenizers raises a plain Exception for a missing file.
        raise DatasetValidationError(f"pinned E5 tokenizer unavailable: {exc}") from exc
    lengths: dict[str, int] = {}
    for record in dataset.corpus:
        text = prepare_memory_text(record.content)
        lengths[f"corpus:{record.memory_key}"] = len(
            tokenizer.encode(text, add_special_tokens=True).ids
        )
    for query in dataset.queries:
        text = ensure_prefix(query.query, ACTIVE_EMBEDDING_PROFILE.query_prefix)
        lengths[f"query:{query.case_id}"] = len(tokenizer.encode(text, add_special_tokens=True).ids)
    overlong = {
        key: value
        for key, value in lengths.items()
        if value > ACTIVE_EMBEDDING_PROFILE.max_input_tokens
    }
    if overlong:
        raise DatasetValidationError(f"E5 input exceeds 512 tokens: {overlong}")
    return lengths


def load_selection_dev(root: Path, *, tokenizer_path: Path | None = None) -> SelectionDev:
    """Читает ровно manifest/corpus/dev; токены проверяются при переданном tokenizer."""

    return _load_selection(root, split="dev", tokenizer_path=tokenizer_path)


def load_selection_validation(root: Path, *, tokenizer_path: Path | None = None) -> SelectionDev:
    """Явный новый validation split; не вызывает старые dataset loaders."""
    return _load_selection(root, split="validation", tokenizer_path=tokenizer_path)


def _load_selection(root, *, split, tokenizer_path):
    root = Path(root)
    query_file = f"{split}.jsonl"
    manifest_class = SelectionManifest if split == "dev" else SelectionValidationManifest
    try:
        manifest = manifest_class.model_validate(
            json.loads(_allowed_path(root, "manifest.json").read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise DatasetValidationError(f"invalid selection manifest: {exc}") from exc
    if set(manifest.files) != {"corpus.jsonl", query_file}:
        raise DatasetValidationError(
            f"selection manifest allows only corpus.jsonl and {query_file}"
        )
    _ensure_unique([source.source_ref for source in manifest.sources], "source_ref")
    try:
        corpus = tuple(
            CorpusRecord.model_validate(value)
            for value in _read_jsonl(
                _allowed_path(root, "corpus.jsonl"), manifest.files["corpus.jsonl"]
            )
        )
        queries = tuple(
            QueryRecord.model_validate(value)
            for value in _read_jsonl(_allowed_path(root, query_file), manifest.files[query_file])
        )
    except ValidationError as exc:
        raise DatasetValidationError(f"invalid selection record: {exc}") from exc
    # _validate_corpus использует только manifest.sources; старый manifest contract не меняем.
    corpus_by_key = _validate_corpus(corpus, cast(DatasetManifest, manifest))
    _ensure_unique([query.case_id for query in queries], "case_id")
    for query in queries:
        _validate_query(query, corpus_by_key)
    if getattr(manifest, split).model_dump() != _computed_split_record(queries):
        raise DatasetValidationError("selection dev summary mismatch")
    if set(manifest.case_rationales) != {query.case_id for query in queries} or any(
        not value.strip() for value in manifest.case_rationales.values()
    ):
        raise DatasetValidationError("case rationales must cover each dev query")
    result = SelectionDev(manifest=manifest, corpus=corpus, queries=queries)
    if tokenizer_path is not None:
        validate_e5_inputs(result, tokenizer_path)
    return result


def validate_independence(dev, validation):
    """Структурная проверка; semantic renaming дополнительно проверяет автор audit."""
    dimensions = {
        "memory_keys": (
            {r.memory_key for r in dev.corpus},
            {r.memory_key for r in validation.corpus},
        ),
        "physical_ids": ({r.id for r in dev.corpus}, {r.id for r in validation.corpus}),
        "logical_ids": (
            {r.logical_id for r in dev.corpus},
            {r.logical_id for r in validation.corpus},
        ),
        "contents": (
            {r.content.casefold().strip() for r in dev.corpus},
            {r.content.casefold().strip() for r in validation.corpus},
        ),
        "source_groups": (
            {r.source_ref for r in dev.corpus},
            {r.source_ref for r in validation.corpus},
        ),
        "paraphrase_families": (
            {q.split_group for q in dev.queries},
            {q.split_group for q in validation.queries},
        ),
        "queries": (
            {q.query.casefold().strip() for q in dev.queries},
            {q.query.casefold().strip() for q in validation.queries},
        ),
    }
    for name, (before, after) in dimensions.items():
        if before & after:
            raise DatasetValidationError(f"validation {name} overlap")
    return {name: True for name in dimensions}
