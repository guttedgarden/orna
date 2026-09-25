"""Контракт frozen selection dev: данные, ссылки и граница закрытых наборов."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from tests.evals.dataset import DatasetValidationError
from tests.evals.experiments.selection_dataset import (
    SelectionDev,
    load_selection_dev,
    validate_e5_inputs,
)

ROOT = Path(__file__).parents[1] / "retrieval" / "selection"


def _copy_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "selection"
    root.mkdir()
    for name in ("manifest.json", "corpus.jsonl", "dev.jsonl"):
        shutil.copyfile(ROOT / name, root / name)
    return root


def _rewrite_jsonl(root: Path, name: str, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (root / name).write_text(payload, encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][name] = {
        "records": len(rows),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _rows(root: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines()]


def test_frozen_dev_has_partial_and_required_slices() -> None:
    dataset = load_selection_dev(ROOT)
    assert dataset.manifest.dataset_version == "selection-dev-v1"
    assert len(dataset.corpus) >= 30
    assert sum(bool(case.relevance) for case in dataset.queries) >= 10
    assert sum("near_topic" in case.slices for case in dataset.queries) >= 10
    assert sum("ood" in case.slices for case in dataset.queries) >= 2
    assert any(1 in case.relevance.values() for case in dataset.queries)
    assert any(len(case.relevance) > 1 for case in dataset.queries)


def test_corrupt_hash_is_rejected(tmp_path: Path) -> None:
    root = _copy_dataset(tmp_path)
    with (root / "corpus.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(DatasetValidationError, match="hash mismatch"):
        load_selection_dev(root)


def test_unknown_relevant_key_is_rejected(tmp_path: Path) -> None:
    root = _copy_dataset(tmp_path)
    rows = _rows(root, "dev.jsonl")
    rows[0]["relevance"] = {"missing-record": 2}
    _rewrite_jsonl(root, "dev.jsonl", rows)
    with pytest.raises(DatasetValidationError, match="unknown relevance key"):
        load_selection_dev(root)


def test_manifest_count_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _copy_dataset(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["dev"]["positive"] -= 1
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="summary mismatch"):
        load_selection_dev(root)


def test_duplicate_case_id_is_rejected(tmp_path: Path) -> None:
    root = _copy_dataset(tmp_path)
    rows = _rows(root, "dev.jsonl")
    rows[1]["case_id"] = rows[0]["case_id"]
    _rewrite_jsonl(root, "dev.jsonl", rows)
    with pytest.raises(DatasetValidationError, match="duplicate case_id"):
        load_selection_dev(root)


@pytest.mark.parametrize(
    ("field", "message"), [("id", "physical id"), ("memory_key", "memory_key")]
)
def test_duplicate_corpus_identity_is_rejected(tmp_path: Path, field: str, message: str) -> None:
    root = _copy_dataset(tmp_path)
    rows = _rows(root, "corpus.jsonl")
    rows[1][field] = rows[0][field]
    _rewrite_jsonl(root, "corpus.jsonl", rows)
    with pytest.raises(DatasetValidationError, match=f"duplicate {message}"):
        load_selection_dev(root)


def test_partial_hint_remains_positive(tmp_path: Path) -> None:
    root = _copy_dataset(tmp_path)
    rows = _rows(root, "dev.jsonl")
    partial = next(row for row in rows if "partial" in row["slices"])
    assert partial["relevance"] and 1 in partial["relevance"].values()
    partial["slices"].append("negative")
    _rewrite_jsonl(root, "dev.jsonl", rows)
    with pytest.raises(DatasetValidationError, match=r"positive query.*negative slice"):
        load_selection_dev(root)


def test_dev_loader_never_opens_closed_files(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.open
    opened: list[str] = []

    def guarded(path: Path, *args: object, **kwargs: object):
        opened.append(path.name)
        assert path.name in {"manifest.json", "corpus.jsonl", "dev.jsonl"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    load_selection_dev(ROOT)
    assert set(opened) == {"manifest.json", "corpus.jsonl", "dev.jsonl"}


def test_manifest_cannot_add_validation_file(tmp_path: Path) -> None:
    root = _copy_dataset(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["validation.jsonl"] = {"records": 1, "sha256": "0" * 64}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=r"only corpus.jsonl and dev.jsonl"):
        load_selection_dev(root)


def test_blank_jsonl_line_is_not_silently_skipped(tmp_path: Path) -> None:
    root = _copy_dataset(tmp_path)
    rows = _rows(root, "dev.jsonl")
    _rewrite_jsonl(root, "dev.jsonl", rows)
    path = root / "dev.jsonl"
    payload = path.read_bytes() + b"\n"
    path.write_bytes(payload)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["dev.jsonl"] = {
        "records": len(rows) + 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="blank line"):
        load_selection_dev(root)


@pytest.mark.parametrize(
    ("case_id", "field", "value", "message"),
    [
        ("p08", "relevance", {"rollout-old": 2}, "not visible"),
        ("p06", "relevance", {"foreign-billing": 2}, "not visible"),
        ("p15", "relevance", {"queue-dead-letter": 2}, "not visible"),
        ("n01", "slices", ["negative"], "inconsistent negative kind"),
    ],
)
def test_lifecycle_visibility_and_type_labels_are_checked(
    tmp_path: Path, case_id: str, field: str, value: object, message: str
) -> None:
    root = _copy_dataset(tmp_path)
    rows = _rows(root, "dev.jsonl")
    row = next(row for row in rows if row["case_id"] == case_id)
    row[field] = value
    if field == "relevance":
        row["forbidden"] = [key for key in row["forbidden"] if key not in value]
    _rewrite_jsonl(root, "dev.jsonl", rows)
    with pytest.raises(DatasetValidationError, match=message):
        load_selection_dev(root)


def test_symlink_to_closed_file_is_rejected_before_read(tmp_path: Path) -> None:
    root = _copy_dataset(tmp_path)
    closed = tmp_path / "validation.jsonl"
    closed.write_text("not to be read", encoding="utf-8")
    (root / "dev.jsonl").unlink()
    (root / "dev.jsonl").symlink_to(closed)
    with pytest.raises(DatasetValidationError, match="inside the dev directory"):
        load_selection_dev(root)


def test_e5_input_guard_counts_full_prefixed_query_without_truncation(tmp_path: Path) -> None:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "query": 1, "x": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    dataset = load_selection_dev(ROOT)
    long_query = dataset.queries[0].model_copy(update={"query": "x " * 520})
    overlong = SelectionDev(dataset.manifest, dataset.corpus, (long_query,))
    with pytest.raises(DatasetValidationError, match="E5 input exceeds 512 tokens"):
        validate_e5_inputs(overlong, tokenizer_path)


def test_missing_pinned_tokenizer_has_explicit_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="pinned E5 tokenizer unavailable"):
        load_selection_dev(ROOT, tokenizer_path=tmp_path / "missing-tokenizer.json")
