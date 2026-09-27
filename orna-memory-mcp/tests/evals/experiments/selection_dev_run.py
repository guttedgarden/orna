"""P2.5-04 diagnostic dev CLI. Ни .env, ни old holdout/validation не читаются."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from math import ceil
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

# app.config имеет module-level Settings(); eval запрещает implicit dotenv I/O.
with patch("pydantic_settings.sources.DotEnvSettingsSource._read_env_files", return_value={}):
    from app.config import Settings

from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE
from app.embeddings import AsyncEmbeddingExecutor, EmbeddingService
from app.model_cache import is_model_cache_ready
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService, reciprocal_rank_fusion
from tests.evals.database import ephemeral_eval_database, inspect_eval_database, load_eval_corpus
from tests.evals.experiments.reranker import (
    Candidate,
    QwenWorkerSession,
    RerankerError,
    rank_candidates,
    score_candidates,
)
from tests.evals.experiments.reranker_selection import analyze_repeats, candidate_admitted
from tests.evals.experiments.selection_dataset import (
    _eligible,
    load_selection_dev,
    load_selection_validation,
    validate_e5_inputs,
    validate_independence,
)
from tests.evals.experiments.selection_embedding import ProcessEmbeddingExecutor
from tests.evals.experiments.selection_resources import ProcessSessionMonitor, SessionMonitor

REPO = Path(__file__).resolve().parents[4]
SERVICE = REPO / "orna-memory-mcp"
DEV = SERVICE / "tests/retrieval/selection"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def percentiles(values):
    values = sorted(values)
    return {
        "n": len(values),
        "p50_seconds": values[ceil(0.5 * len(values)) - 1] if values else None,
        "p95_seconds": values[ceil(0.95 * len(values)) - 1] if values else None,
    }


def identity(settings):
    files = [
        *sorted((SERVICE / "app").glob("*.py")),
        *sorted((SERVICE / "app/schema/migrations").glob("*.sql")),
        *sorted(Path(__file__).parent.glob("*.py")),
        *sorted((Path(__file__).parent / "qwen_runtime").glob("*.py")),
        *sorted((SERVICE / "tests/evals").glob("*.py")),
        SERVICE / "tests/unit/test_selection_dev_experiment.py",
        Path(__file__).parent / "qwen_runtime/pyproject.toml",
        SERVICE / "uv.lock",
        SERVICE / "pyproject.toml",
        Path(__file__).parent / "qwen_runtime/uv.lock",
        DEV / "README.md",
        DEV / "manifest.json",
        DEV / "corpus.jsonl",
        DEV / "dev.jsonl",
        SERVICE / "tests/retrieval/baseline.json",
        REPO / "docs/branch-plan/phase-2-5-memory-selection.md",
    ]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args])

    return {
        "git_head": git("rev-parse", "HEAD").decode().strip(),
        "git_status": git("status", "--porcelain").decode(),
        "dirty_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
        "files_sha256": {str(p.relative_to(REPO)): digest(p) for p in files},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "e5_profile": asdict(ACTIVE_EMBEDDING_PROFILE),
        "e5_files_sha256": {
            name: digest(
                ACTIVE_EMBEDDING_PROFILE.snapshot_path(settings.embedding_cache_dir) / name
            )
            for name in ACTIVE_EMBEDDING_PROFILE.required_files
        },
    }


class CaptureRepository:
    """Делегирует реальные SQL; сохраняет ровно channels вызова service baseline."""

    def __init__(self, repository):
        self.repository = repository
        self.channels = {}

    async def search_dense(self, *args, **kwargs):
        result = await self.repository.search_dense(*args, **kwargs)
        self.channels["dense"] = result
        return result

    async def search_lexical(self, *args, **kwargs):
        result = await self.repository.search_lexical(*args, **kwargs)
        self.channels["lexical"] = result
        return result


def container_footprint(container):
    try:
        row = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", container],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return json.loads(row.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"measurement": None, "status": "unknown"}


def same_identity(actual, frozen):
    return json.dumps(actual, sort_keys=True) == json.dumps(frozen, sort_keys=True)


def load_research_candidate(path, expected_sha256):
    if digest(path) != expected_sha256:
        raise ValueError("frozen candidate hash mismatch")
    candidate = json.loads(path.read_text())
    anchor = json.loads((DEV / "research-freeze-v2.json").read_text())
    if (
        expected_sha256 != anchor["candidate_sha256"]
        or candidate["threshold"] != anchor["threshold"]
    ):
        raise ValueError("independent freeze anchor mismatch")
    if (
        candidate["replay_sha256"] != anchor["replay_sha256"]
        or digest(path.with_name("replay.json")) != anchor["replay_sha256"]
    ):
        raise ValueError("frozen replay hash mismatch")
    if (
        candidate.get("artifact_type") != "research-only-reranker"
        or candidate.get("policy") != "selection-protocol-v2"
        or candidate.get("research_only") is not True
        or candidate.get("ready_for_research_validation") is not True
        or candidate.get("production_admitted") is not False
    ):
        raise ValueError("invalid research candidate")
    threshold = candidate["threshold"]
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("invalid frozen threshold")
    if digest(DEV / "protocol-v2.md") != candidate["policy_sha256"]:
        raise ValueError("policy hash mismatch")
    for name, expected in candidate["inference_files_sha256"].items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO) or digest(path) != expected:
            raise ValueError("frozen inference code/config hash mismatch")
    return candidate


async def run(args):
    validation = getattr(args, "validation", None)
    candidate = None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings(
        _env_file=None,
        postgres_host="127.0.0.1",
        postgres_port=args.port,
        postgres_user="orna",
        postgres_password="",
        postgres_db="template1",
        database_url=None,
        embedding_cache_dir=REPO / "data/fastembed",
        embedding_local_files_only=True,
        dense_retrieval_strategy="exact",
        retrieval_candidate_pool_size=20,
        rrf_k=60,
    )
    artifact = {
        "experiment": "P2.5-05-research-validation" if validation else "P2.5-04",
        "diagnostic_exception": "user-2026-09-26",
        "status": "partial",
        "historical_p2_5_03_resource_gate": False,
        "resource_policy": "host-swap-diagnostic-v1",
        "host_swap_role": "diagnostic_only",
        "repeats": [],
        "warmup": [],
        "config": {"seed": 0, "pool": 20, "rrf_k": 60, "cutoff": 5, "warmups": 2, "repeats": 3},
        "candidate_admitted": False,
        "results_used": None,
        "task_success": None,
        "client_prompt_tokens": None,
    }
    client = QwenWorkerSession(REPO / "data/models/phase-2-5/hub")
    monitor_class = ProcessSessionMonitor if validation else SessionMonitor
    monitor = monitor_class(output / "resources.jsonl", worker=lambda: client.process)
    client.check_budget = monitor.check
    events = (output / "queries.jsonl").open("x")

    def event(row):
        events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        events.flush()

    try:
        if not is_model_cache_ready(settings.embedding_cache_dir):
            raise RuntimeError("pinned offline E5 cache unavailable")
        if validation:
            candidate = load_research_candidate(args.candidate, args.candidate_sha256)
            artifact["frozen_candidate_sha256"] = args.candidate_sha256
            artifact["frozen_threshold"] = candidate["threshold"]
            artifact["policy"] = candidate["policy"]
        loader = load_selection_validation if validation else load_selection_dev
        dataset = loader(
            validation if validation else DEV,
            tokenizer_path=ACTIVE_EMBEDDING_PROFILE.snapshot_path(settings.embedding_cache_dir)
            / "tokenizer.json",
        )
        artifact["e5_input_lengths"] = validate_e5_inputs(
            dataset,
            ACTIVE_EMBEDDING_PROFILE.snapshot_path(settings.embedding_cache_dir) / "tokenizer.json",
        )
        artifact["identity"] = identity(settings)
        if candidate:
            for key in ("e5_profile", "e5_files_sha256", "packages", "python"):
                if not same_identity(artifact["identity"][key], candidate["dev_identity"][key]):
                    raise ValueError("frozen E5/core runtime identity mismatch")
            if artifact["config"] != candidate["config"]:
                raise ValueError("frozen retrieval config mismatch")
        artifact["dataset"] = dataset.manifest.model_dump(mode="json")
        if validation:
            artifact["independence_checks"] = validate_independence(
                load_selection_dev(DEV), dataset
            )
            frozen = {
                name: digest(validation / name)
                for name in ("manifest.json", "corpus.jsonl", "validation.jsonl")
            }
            artifact["validation_files_sha256"] = frozen
            audit = json.loads((validation / "independence-audit.json").read_text())
            if (
                audit["files_sha256"] != frozen
                or audit["candidate_sha256"] != args.candidate_sha256
            ):
                raise ValueError("validation independence audit hash mismatch")
            artifact["independence_audit_sha256"] = digest(validation / "independence-audit.json")
            dump(
                output / "pre-inference-freeze.json",
                {
                    "candidate_sha256": args.candidate_sha256,
                    "dataset_sha256": frozen,
                    "code_identity": artifact["identity"],
                    "model_inference_started": False,
                },
            )
        queries = sorted(dataset.queries, key=lambda q: q.case_id)
        random.Random(0).shuffle(queries)
        artifact["query_order"] = [q.case_id for q in queries]
        query_specs = [
            dict(q.model_dump(), allowed=[r.memory_key for r in dataset.corpus if _eligible(r, q)])
            for q in queries
        ]
        artifact["query_specs"] = query_specs
        key_by_id = {str(r.id): r.memory_key for r in dataset.corpus}
        artifact["db_container_before"] = container_footprint(args.container)
        dump(output / "run.json", artifact)
        with monitor:
            setup = perf_counter()
            async with ephemeral_eval_database(settings) as db:
                artifact["database_name"] = db.name
                embeddings = (
                    ProcessEmbeddingExecutor(db.settings, monitor)
                    if validation
                    else AsyncEmbeddingExecutor(EmbeddingService(db.settings))
                )
                try:
                    monitor.phase = "e5-load"
                    load_started = perf_counter()
                    await asyncio.wait_for(
                        embeddings.embed_query("Which port does the demo service use?"), timeout=120
                    )
                    artifact["e5_cold_load_and_first_query_seconds"] = perf_counter() - load_started
                    monitor.check()
                    monitor.phase = "corpus-build"
                    await load_eval_corpus(db.pool, dataset.corpus, embeddings, db.settings)
                    monitor.check()
                    artifact["database"], artifact["corpus"] = await inspect_eval_database(db.pool)
                    artifact["provisioning_seconds"] = perf_counter() - setup
                    repository = CaptureRepository(MemoryRepository(db.pool, db.settings))
                    service = MemorySearchService(repository, embeddings, db.settings)
                    monitor.phase = "qwen-load"
                    monitor.deadline = perf_counter() + 7200
                    session_start = perf_counter()
                    with client:
                        artifact["qwen"] = client.ready
                        if candidate and any(
                            client.ready.get(k) != v for k, v in candidate["qwen"].items()
                        ):
                            raise RuntimeError("frozen Qwen identity mismatch")
                        artifact["worker_process_startup_seconds"] = perf_counter() - session_start
                        sanity = [
                            Candidate("good", "good", "The demo service listens on port 8123."),
                            Candidate("bad", "bad", "The garden has three apple trees."),
                        ]
                        for i in range(3):
                            monitor.phase = "process-cold-first-query" if i == 0 else f"warmup-{i}"
                            monitor.query_deadline = perf_counter() + 180
                            started = perf_counter()
                            rows = score_candidates(
                                f"sanity-{i}",
                                "Which port does the demo service use?",
                                sanity,
                                client,
                            )
                            if rows[0].score <= rows[1].score:
                                raise RuntimeError("sanity ordering failed")
                            artifact["warmup"].append(
                                {
                                    "iteration": i,
                                    "seconds": perf_counter() - started,
                                    "scores": [asdict(x) for x in rows],
                                }
                            )
                            monitor.check()
                            monitor.query_deadline = None
                        for repetition in range(3):
                            samples = []
                            artifact["repeats"].append(samples)
                            for query in queries:
                                monitor.phase = f"repeat-{repetition + 1}/{query.case_id}"
                                monitor.query_deadline = perf_counter() + 180
                                started = perf_counter()
                                baseline = await asyncio.wait_for(
                                    service.search(
                                        MemorySearchQuery(
                                            query=query.query,
                                            memory_type=query.memory_type,
                                            limit=5,
                                        ),
                                        query.project_id,
                                    ),
                                    timeout=180,
                                )
                                pool = reciprocal_rank_fusion(
                                    repository.channels["dense"],
                                    repository.channels["lexical"],
                                    k=60,
                                    limit=20,
                                )
                                if [r.id for r in baseline] != [r.id for r in pool[:5]]:
                                    raise RuntimeError("service top5 differs from fused pool")
                                for r in pool:
                                    if key_by_id[str(r.id)] not in next(
                                        q["allowed"]
                                        for q in query_specs
                                        if q["case_id"] == query.case_id
                                    ):
                                        raise RuntimeError("visibility/lifecycle/type failure")
                                retrieval_seconds = perf_counter() - started
                                sample = {
                                    "case_id": query.case_id,
                                    "repeat": repetition + 1,
                                    "baseline": [key_by_id[str(r.id)] for r in baseline],
                                    "pool": [
                                        dict(r.model_dump(mode="json"), key=key_by_id[str(r.id)])
                                        for r in pool
                                    ],
                                    "channels": {
                                        name: [
                                            {
                                                "id": str(r.id),
                                                "key": key_by_id[str(r.id)],
                                                "position": j,
                                                "score": score,
                                            }
                                            for j, (r, score) in enumerate(channel)
                                        ]
                                        for name, channel in repository.channels.items()
                                    },
                                }
                                event({"event": "retrieval", **sample})
                                stage_start = perf_counter()
                                scored = score_candidates(
                                    query.case_id,
                                    query.query,
                                    [
                                        Candidate(str(r.id), str(r.logical_id), r.content)
                                        for r in pool
                                    ],
                                    client,
                                )
                                ranked = rank_candidates(scored)
                                # Все возможные query-level filter outputs измеряются сейчас;
                                # threshold selection позже только replay, без model calls.
                                thresholds = (
                                    [candidate["threshold"]]
                                    if candidate
                                    else {0.0, 1.0, *(r.score for r in ranked)}
                                )
                                for threshold in thresholds:
                                    [r.id for r in ranked if r.score >= threshold][:5]
                                sample["scores"] = [
                                    dict(asdict(r), key=key_by_id[r.id]) for r in scored
                                ]
                                sample["reranker_seconds"] = perf_counter() - stage_start
                                sample["retrieval_seconds"] = retrieval_seconds
                                sample["pipeline_seconds"] = (
                                    retrieval_seconds + sample["reranker_seconds"]
                                )
                                sample["query_wall_seconds"] = perf_counter() - started
                                relevant = set(query.relevance)
                                pool_keys = {r["key"] for r in sample["pool"]}
                                sample["candidate_recall_at_20"] = (
                                    len(relevant & pool_keys) / len(relevant) if relevant else None
                                )
                                sample["candidate_misses"] = sorted(relevant - pool_keys)
                                samples.append(sample)
                                event({"event": "complete", **sample})
                                monitor.check()
                                monitor.query_deadline = None
                                dump(output / "run.json", artifact)
                                print(
                                    json.dumps(
                                        {
                                            "repeat": repetition + 1,
                                            "case_id": query.case_id,
                                            "pairs": len(scored),
                                            "seconds": round(sample["pipeline_seconds"], 3),
                                        }
                                    ),
                                    flush=True,
                                )
                            for key, budget in (("reranker_seconds", 60), ("pipeline_seconds", 65)):
                                if percentiles([s[key] for s in samples])["p95_seconds"] > budget:
                                    raise RuntimeError(f"{key} p95 budget exceeded")
                        artifact["quality_run_seconds"] = perf_counter() - session_start
                finally:
                    await embeddings.aclose()
        if validation and (
            any(digest(validation / name) != h for name, h in frozen.items())
            or digest(args.candidate) != args.candidate_sha256
        ):
            raise RuntimeError("frozen validation inputs changed during run")
        artifact["status"] = "complete-diagnostic"
    except Exception as exc:
        # Не сериализуем DSN/credentials из сторонних exceptions.
        artifact["failure"] = {"type": type(exc).__name__, "phase": monitor.phase}
        if isinstance(exc, RerankerError) or (
            isinstance(exc, RuntimeError) and type(exc).__module__ in (__name__, "builtins")
        ):
            artifact["failure"]["reason"] = str(exc)
    finally:
        events.close()
        artifact["resources"] = monitor.summary()
        artifact["resources"]["worker_lifetime_high_water_bytes"] = client.peak_rss_bytes or None
        artifact["db_container_after"] = container_footprint(args.container)
        complete = artifact["status"] == "complete-diagnostic"
        if complete:
            try:
                analysis = analyze_repeats(
                    query_specs,
                    artifact["repeats"],
                    policy="selection-protocol-v2" if candidate else "selection-protocol-v1",
                    fixed_threshold=candidate["threshold"] if candidate else None,
                )
                dump(
                    output / ("validation-results.json" if candidate else "threshold-trials.json"),
                    analysis,
                )
                if candidate:
                    artifact["quality_validation_pass"] = analysis["trials"][0]["quality_pass"]
                artifact["quality_threshold"] = analysis["quality_threshold"]
            except Exception as exc:
                complete = False
                artifact["status"] = "analysis-failed"
                artifact["failure"] = {"type": type(exc).__name__, "phase": "score-replay"}
        for key in (
            "retrieval_seconds",
            "reranker_seconds",
            "pipeline_seconds",
            "query_wall_seconds",
        ):
            artifact.setdefault("timings", {})[key] = {
                "per_repeat": [percentiles([s[key] for s in r]) for r in artifact["repeats"]],
                "pooled": percentiles([s[key] for r in artifact["repeats"] for s in r]),
            }
        resource = artifact["resources"]
        gates = {
            "complete": complete,
            "qwen_rss": (client.peak_rss_bytes <= 6 * 1024**3)
            if client.peak_rss_bytes
            else resource["worker_rss_pass"],
            "aggregate_rss": resource["aggregate_rss_pass"],
            "query_timeout": complete,
            "qwen_load": artifact.get("qwen", {}).get("load_seconds", float("inf")) <= 120,
            "e5_load": artifact.get("e5_cold_load_and_first_query_seconds", float("inf")) <= 120,
            "full_run": artifact.get("quality_run_seconds", float("inf")) <= 7200,
        }
        for key, budget in (("reranker_seconds", 60), ("pipeline_seconds", 65)):
            series = artifact["timings"][key]
            gates[key] = complete and all(
                x["n"] >= 30 and x["p95_seconds"] <= budget
                for x in [*series["per_repeat"], series["pooled"]]
            )
        artifact["cost_gates"] = gates
        validation_pass = artifact.get("quality_validation_pass", False)
        artifact["admission_gates"] = dict(gates, independent_quality_validation=validation_pass)
        artifact["candidate_admitted"] = candidate_admitted(
            artifact.get("quality_threshold"), gates, validation_pass=validation_pass
        )
        dump(output / "run.json", artifact)
    return 0 if artifact["status"] == "complete-diagnostic" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    args = parser.parse_args()
    if bool(args.validation) != bool(args.candidate) or bool(args.candidate) != bool(
        args.candidate_sha256
    ):
        parser.error("validation requires candidate and candidate-sha256 together")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
