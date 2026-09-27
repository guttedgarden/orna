"""Проверка неизменного P2.5-04 и research-only freeze без inference."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from tests.evals.experiments.reranker_selection import analyze_repeats
from tests.evals.experiments.selection_dataset import _eligible, load_selection_dev

SERVICE = Path(__file__).resolve().parents[3]
REPO = SERVICE.parent
DEV = SERVICE / "tests/retrieval/selection"
EXPECTED = {
    "queries.jsonl": "47fc61ae7b0bb5adee108e680f1c76267fd0a6f662211e1ce56d141c0ed5ceff",
    "resource-windows.json": "1630cd9ca662ff48c4a82f556825f7e25153f1f57943359fe0a4ea46f18c680c",
    "resources.jsonl": "66e1f5ec29b80e159b250973502e80bab766e3fce835fc8dc9bc515a459bf1ec",
    "run.json": "584c2f5232ca25122803ba4277d16a3467227a9a75446cf0a943bc2158d6222a",
    "threshold-trials.json": "89ac376a223b3752cd4199d0e1810b2abd4085f67d1c7a66926772e75897eeaf",
}


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def replay(source, output):
    hashes = {name: sha(source / name) for name in EXPECTED}
    if hashes != EXPECTED:
        raise ValueError("historical artifact hash mismatch")
    old = json.loads((source / "run.json").read_text())
    dataset = load_selection_dev(DEV)
    if old["dataset"] != dataset.manifest.model_dump(mode="json"):
        raise ValueError("dev identity mismatch")
    if old["status"] != "complete-diagnostic" or len(old["repeats"]) != 3:
        raise ValueError("incomplete historical run")
    queries = [
        dict(q.model_dump(), allowed=[r.memory_key for r in dataset.corpus if _eligible(r, q)])
        for q in dataset.queries
    ]
    queries.sort(key=lambda q: old["query_order"].index(q["case_id"]))
    events = [json.loads(line) for line in (source / "queries.jsonl").read_text().splitlines()]
    completed = [
        {k: v for k, v in e.items() if k != "event"} for e in events if e["event"] == "complete"
    ]
    if completed != [s for repeat in old["repeats"] for s in repeat]:
        raise ValueError("incremental event mismatch")
    historical = analyze_repeats(queries, old["repeats"], policy="selection-protocol-v1")
    if historical != json.loads((source / "threshold-trials.json").read_text()):
        raise ValueError("historical v1 replay mismatch")
    current = analyze_repeats(queries, old["repeats"])
    if [t["threshold"] for t in current["trials"]] != [
        t["threshold"] for t in historical["trials"]
    ]:
        raise ValueError("threshold grid changed")
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "replay.json",
        {
            "policy": "selection-protocol-v2",
            "source_sha256": hashes,
            "historical_v1_reproduced": True,
            "analysis": current,
        },
    )
    threshold = current["quality_threshold"]
    if threshold is None:
        return None
    inference_files = {
        p: h
        for p, h in old["identity"]["files_sha256"].items()
        if "/app/" in p
        or p.endswith(
            (
                "reranker.py",
                "qwen_runtime/worker.py",
                "qwen_runtime/uv.lock",
                "qwen_runtime/pyproject.toml",
                "baseline.json",
                "orna-memory-mcp/uv.lock",
                "orna-memory-mcp/pyproject.toml",
            )
        )
    }
    if any(sha(REPO / p) != h for p, h in inference_files.items()):
        raise ValueError("inference code/config changed since dev")
    candidate = {
        "artifact_type": "research-only-reranker",
        "policy": "selection-protocol-v2",
        "frozen_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "ready_for_research_validation": True,
        # Dev freeze не является независимой validation независимо от cost.
        "production_admitted": False,
        "admission_reason": "independent_validation_required",
        "resource_policy": "host-swap-diagnostic-v1",
        "host_swap_role": "diagnostic_only",
        "threshold": threshold,
        "comparison": "score >= threshold",
        "candidate_policy": (
            "exact dense20 + FTS simple20; RRF k60 union capped20; filtered top5; no refill"
        ),
        "ties": "original pool position, logical_id, id",
        "truncation": "head-only body within 2048 including prefix/suffix; no query truncation",
        "qwen": {
            k: v
            for k, v in old["qwen"].items()
            if k not in {"pid", "ready", "load_seconds", "worker_peak_rss_bytes"}
        },
        "config": old["config"],
        "dev_identity": old["identity"],
        "dataset": old["dataset"],
        "inference_files_sha256": inference_files,
        "policy_sha256": sha(DEV / "protocol-v2.md"),
        "replay_code_sha256": {
            p.name: sha(p)
            for p in [Path(__file__), Path(__file__).with_name("reranker_selection.py")]
        },
        "replay_sha256": sha(output / "replay.json"),
        "dev_results": {
            "baseline": current["baseline"],
            "rejection": current["rejection"],
            "eligible_trials": sum(t["quality_pass"] for t in current["trials"]),
            "total_trials": len(current["trials"]),
        },
        "historical_cost_gates": old["cost_gates"],
        "historical_p2_5_03_resource_gate": False,
        "historical_resources": old["resources"],
        "source_sha256": hashes,
    }
    write_new(output / "research-candidate.json", candidate)
    return candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = replay(args.source, args.output)
    print(
        json.dumps(
            None
            if candidate is None
            else {
                "threshold": candidate["threshold"],
                "eligible_trials": candidate["dev_results"]["eligible_trials"],
                "production_admitted": candidate["production_admitted"],
            }
        )
    )
