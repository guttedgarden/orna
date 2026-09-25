"""Dev-only калибровка отсева после production RRF, без нового inference на каждый порог."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.config import Settings
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE
from app.embeddings import AsyncEmbeddingExecutor, EmbeddingService
from app.model_cache import is_model_cache_ready
from app.normalizer import normalize_query_to_lexical_groups
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService, reciprocal_rank_fusion
from tests.evals.database import ephemeral_eval_database, inspect_eval_database, load_eval_corpus
from tests.evals.dataset import (
    CorpusRecord,
    QueryRecord,
    SourceRecord,
    _validate_corpus,
    _validate_queries,
    load_retrieval_split,
)
from tests.evals.experiments.run import _metrics

ROOT = Path(__file__).resolve().parents[2] / "retrieval"
OOD_CASE_IDS = frozenset({"abst-dev-unrelated", "val-ood-weather", "val-ood-chess"})


def validation_outcome(baseline: dict[str, Any], trial: dict[str, Any]) -> dict[str, bool]:
    """Улучшение только OOD не объявляется решением near-topic задачи."""
    numeric = not trial["regressions"] and trial["negative_hits"] < baseline["negative_hits"]
    near_topic = trial["near_topic_negative_hits"] < baseline["near_topic_negative_hits"]
    return {
        "numeric_gate_passed": numeric,
        "near_topic_gain": near_topic,
        "supports_near_topic_fix": numeric and near_topic,
        "runtime_promotion": False,
    }


def filter_ranking(rows: list[dict[str, Any]], rule: str, threshold: float | None) -> list[str]:
    """Фильтрует final top-5, сохраняя порядок; lexical-only hit не имеет dense score."""
    if rule not in {"cosine", "cosine_or_lexical", "lexical"}:
        raise ValueError("unknown abstention rule")
    if rule != "lexical" and (
        threshold is None or not math.isfinite(threshold) or not -1 <= threshold <= 1
    ):
        raise ValueError("cosine threshold must be finite and between -1 and 1")
    kept = []
    for row in rows:
        similarity = row["similarity"]
        if similarity is not None and not math.isfinite(similarity):
            raise ValueError("non-finite similarity")
        lexical = row["lexical"] and rule in {"lexical", "cosine_or_lexical"}
        dense = rule != "lexical" and similarity is not None and similarity >= threshold
        if lexical or dense:
            kept.append(row["key"])
    return kept


def regression_cases(cases: list[dict[str, Any]], rankings: dict[str, list[str]]) -> list[str]:
    """Не допускает потери отдельных relevant hits или уже успешного top-1."""
    failures = []
    for case in cases:
        baseline = case["baseline"]
        ranking = rankings[case["case_id"]]
        relevant = set(case["relevance"])
        lost = (relevant & set(baseline)) - set(ranking)
        lost_top1 = baseline and baseline[0] in relevant and ranking[:1] != baseline[:1]
        if lost or lost_top1:
            failures.append(case["case_id"])
    return failures


def select_candidate(
    trials: list[dict[str, Any]],
    *,
    baseline_negative_hits: int,
    baseline_new_negative_hits: int | None = None,
) -> dict[str, Any] | None:
    """Только dev-кандидат: zero regressions и строго меньше negative hits."""
    eligible = [
        trial
        for trial in trials
        if not trial["regressions"]
        and trial["negative_hits"] < baseline_negative_hits
        and (
            baseline_new_negative_hits is None
            or trial["new_negative_hits"] < baseline_new_negative_hits
        )
    ]
    return min(
        eligible,
        key=lambda t: (t["negative_hits"], t["returned"], t["rule"], t["threshold"] or 0),
        default=None,
    )


def load_dev_cases(root: Path = ROOT):
    dataset = load_retrieval_split(root, "dev")
    extra = tuple(
        QueryRecord.model_validate_json(line)
        for line in (root / "abstention_dev.jsonl").read_text().splitlines()
        if line.strip()
    )
    queries = (*dataset.queries, *extra)
    _validate_queries(queries, (), {record.memory_key: record for record in dataset.corpus})
    return dataset, queries


def load_validation_cases(root: Path = ROOT):
    dataset, dev = load_dev_cases(root)
    added = tuple(
        CorpusRecord.model_validate_json(line)
        for line in (root / "abstention_validation_corpus.jsonl").read_text().splitlines()
        if line.strip()
    )
    queries = tuple(
        QueryRecord.model_validate_json(line)
        for line in (root / "abstention_validation.jsonl").read_text().splitlines()
        if line.strip()
    )
    manifest = dataset.manifest.model_copy(
        update={
            "sources": [
                *dataset.manifest.sources,
                SourceRecord(
                    source_ref="synthetic-abstention-validation-v1",
                    description="Independent synthetic engineering fixtures; not live facts",
                ),
            ]
        }
    )
    corpus = (*dataset.corpus, *added)
    by_key = _validate_corpus(corpus, manifest)
    _validate_queries(dev, queries, by_key)
    return dataset.model_copy(update={"corpus": corpus}), queries


async def run(output: Path, *, validation: bool = False) -> Path:
    # Артефакт не перезаписывается даже после неудачного запуска.
    output.mkdir(parents=True, exist_ok=False)
    dataset, queries = load_validation_cases() if validation else load_dev_cases()
    candidate = json.loads((ROOT / "abstention_candidate.json").read_text()) if validation else None
    if candidate and (
        candidate["embedding_profile"] != ACTIVE_EMBEDDING_PROFILE.version
        or candidate["embedding_snapshot"] != ACTIVE_EMBEDDING_PROFILE.source_revision
    ):
        raise ValueError("candidate embedding profile mismatch")
    settings = Settings().model_copy(
        update={
            "embedding_local_files_only": True,
            "dense_retrieval_strategy": "exact",
            "retrieval_candidate_pool_size": 20,
            "rrf_k": 60,
        }
    )
    if override := os.environ.get("ORNA_TEST_E5_CACHE_DIR"):
        settings = settings.model_copy(update={"embedding_cache_dir": Path(override)})
    if not is_model_cache_ready(settings.embedding_cache_dir):
        raise RuntimeError("real pinned E5 offline cache is required")
    key_by_id = {record.id: record.memory_key for record in dataset.corpus}
    cases = []
    async with ephemeral_eval_database(settings) as db:
        embeddings = AsyncEmbeddingExecutor(
            EmbeddingService(db.settings), max_concurrency=settings.embedding_max_concurrency
        )
        try:
            await load_eval_corpus(db.pool, dataset.corpus, embeddings, db.settings)
            repository = MemoryRepository(db.pool, db.settings)
            service = MemorySearchService(repository, embeddings, db.settings)
            for query in queries:
                started = perf_counter()
                vector = await embeddings.embed_query(query.query)
                embedding_ms = (perf_counter() - started) * 1000
                samples = []
                rows = []
                for _ in range(3):
                    started = perf_counter()
                    dense, lexical = await asyncio.gather(
                        repository.search_dense(
                            vector,
                            query.project_id,
                            20,
                            strategy="exact",
                            memory_type=query.memory_type,
                        ),
                        repository.search_lexical(
                            normalize_query_to_lexical_groups(query.query),
                            query.project_id,
                            20,
                            memory_type=query.memory_type,
                        ),
                    )
                    fused = reciprocal_rank_fusion(dense, lexical, k=60, limit=5)
                    current = [
                        {
                            "key": key_by_id[r.id],
                            "similarity": None if r.distance is None else 1 - r.distance,
                            "lexical": r.rank_lexical is not None,
                            "rrf_score": r.rrf_score,
                        }
                        for r in fused
                    ]
                    if rows and current != rows:
                        raise RuntimeError("unstable score/ranking across exact repeats")
                    rows = current
                    samples.append((perf_counter() - started) * 1000)
                # Реальный application boundary должен совпасть с диагностикой repository+RRF.
                actual = await service.search(
                    MemorySearchQuery(query=query.query, memory_type=query.memory_type),
                    query.project_id,
                )
                baseline = [row["key"] for row in rows]
                if [key_by_id[r.id] for r in actual] != baseline:
                    raise RuntimeError("diagnostic ranking differs from production service")
                cases.append(
                    {
                        "case_id": query.case_id,
                        "query": query.query,
                        "relevance": query.relevance,
                        "slices": query.slices,
                        "suite": "validation"
                        if validation
                        else ("new_dev" if query.case_id.startswith("abst-dev-") else "frozen_dev"),
                        "baseline": baseline,
                        "rows": rows,
                        "embedding_ms": embedding_ms,
                        "retrieval_fusion_ms": samples,
                    }
                )
            database, corpus_info = await inspect_eval_database(db.pool)
        finally:
            await embeddings.aclose()

    def measure(rule, threshold):
        rankings = {
            c["case_id"]: c["baseline"]
            if rule == "baseline"
            else filter_ranking(c["rows"], rule, threshold)
            for c in cases
        }
        return {
            "rule": rule,
            "threshold": threshold,
            "regressions": regression_cases(cases, rankings),
            "negative_hits": sum(bool(rankings[c["case_id"]]) for c in cases if not c["relevance"]),
            "near_topic_negative_hits": sum(
                bool(rankings[c["case_id"]])
                for c in cases
                if not c["relevance"] and c["case_id"] not in OOD_CASE_IDS
            ),
            "ood_negative_hits": sum(
                bool(rankings[c["case_id"]])
                for c in cases
                if not c["relevance"] and c["case_id"] in OOD_CASE_IDS
            ),
            "new_negative_hits": sum(
                bool(rankings[c["case_id"]])
                for c in cases
                if not c["relevance"] and c["suite"] == "new_dev"
            )
            if not validation
            else None,
            "returned": sum(map(len, rankings.values())),
            "metrics": _metrics(dataset.corpus, queries, rankings),
            "suite_metrics": {
                name: _metrics(
                    dataset.corpus, tuple(q for q in queries if q.case_id in ids), rankings
                )
                for name, ids in (
                    (name, {c["case_id"] for c in cases if c["suite"] == name})
                    for name in sorted({c["suite"] for c in cases})
                )
            },
            "rankings": rankings,
        }

    baseline = measure("baseline", None)
    if candidate:
        # Единственный заранее сохранённый вариант; validation не подбирает пороги.
        trials = [measure(candidate["rule"], candidate["threshold"])]
        selected = None
    else:
        # Все варианты отсева в разрешённом диапазоне [-1, 1], только на dev.
        thresholds = sorted(
            {
                -1.0,
                1.0,
                *(
                    row["similarity"]
                    for case in cases
                    for row in case["rows"]
                    if row["similarity"] is not None
                ),
            }
        )
        trials = [
            measure(rule, threshold)
            for rule in ("cosine", "cosine_or_lexical")
            for threshold in thresholds
        ]
        trials.append(measure("lexical", None))
        selected = select_candidate(
            trials,
            baseline_negative_hits=baseline["negative_hits"],
            baseline_new_negative_hits=baseline["new_negative_hits"],
        )
    repo_root = Path(__file__).resolve().parents[4]
    files = [ROOT / name for name in ("corpus.jsonl", "memory_dev.jsonl", "abstention_dev.jsonl")]
    if validation:
        files += [
            ROOT / name
            for name in (
                "abstention_validation_corpus.jsonl",
                "abstention_validation.jsonl",
                "abstention_candidate.json",
            )
        ]
    measured = [
        Path(__file__),
        repo_root / "orna-memory-mcp/tests/unit/test_abstention_experiment.py",
        repo_root / "orna-memory-mcp/app/search.py",
        repo_root / "orna-memory-mcp/app/repository.py",
    ]
    artifact = {
        "experiment": "abstention-validation-v1" if validation else "abstention-dev-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "file_sha256": {
            str(p.relative_to(repo_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*files, *measured]
        },
        "embedding_profile": ACTIVE_EMBEDDING_PROFILE.version,
        "embedding_snapshot": ACTIVE_EMBEDDING_PROFILE.source_revision,
        "config": {
            "strategy": "exact",
            "pool": 20,
            "rrf_k": 60,
            "cutoff": 5,
            "fts": "simple",
            "repeats": 3,
            "postfilter": "no refill",
        },
        "database": database,
        "corpus": corpus_info,
        "cases": cases,
        "baseline": baseline,
        "trials": trials,
        "selected_dev_candidate": selected,
        "validation_candidate": candidate,
        "validation_outcome": validation_outcome(baseline, trials[0]) if validation else None,
        "new_independent_validation_run": validation,
        "old_phase2_holdout_run": False,
        "old_holdout_read": False,
        "production_changed": False,
    }
    path = output / "run.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Abstention validation" if validation else "# Abstention dev experiment",
        "",
        "| Rule | Threshold | Negative hits | Regressions | Returned |",
        "|---|---:|---:|---:|---:|",
    ]
    safe = [t for t in trials if not t["regressions"]]
    best_safe = min(safe, key=lambda t: (t["negative_hits"], t["returned"]), default=None)
    displayed = (
        [baseline, trials[0]]
        if validation
        else [baseline, *([best_safe] if best_safe else []), trials[-1]]
    )
    for t in displayed:
        lines.append(
            f"| {t['rule']} | {t['threshold']} | {t['negative_hits']} | "
            f"{len(t['regressions'])} | {t['returned']} |"
        )
    lines += [
        "",
        "Selected dev candidate: "
        f"{None if selected is None else (selected['rule'], selected['threshold'])}",
        f"New validation run: {validation}. Old holdout not read. No production change.",
        f"Validation outcome: {artifact['validation_outcome']}. Details: run.json.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-candidate", action="store_true")
    args = parser.parse_args()
    print(asyncio.run(run(args.output, validation=args.validate_candidate)))
