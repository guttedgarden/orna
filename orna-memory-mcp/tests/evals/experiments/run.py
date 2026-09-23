"""CLI P2-04: real-E5 dev-only controlled experiments in one owned eval DB."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from app.config import Settings
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE
from app.embeddings import AsyncEmbeddingExecutor, EmbeddingService
from app.normalizer import normalize_query_to_lexical_groups
from app.repository import MemoryRepository
from app.search import reciprocal_rank_fusion
from tests.evals.database import ephemeral_eval_database, inspect_eval_database, load_eval_corpus
from tests.evals.dataset import CorpusRecord, QueryRecord, load_retrieval_split
from tests.evals.experiments.ann import run_ann
from tests.evals.experiments.lexical import run_lexical
from tests.evals.experiments.trigram import run_trigram
from tests.evals.metrics import MetricCase, evaluate_rankings

RETRIEVAL_ROOT = Path(__file__).resolve().parents[2] / "retrieval"


def _metrics(
    corpus: tuple[CorpusRecord, ...],
    queries: tuple[QueryRecord, ...],
    rankings: dict[str, list[str]],
) -> dict[str, Any]:
    summary = evaluate_rankings(
        tuple(
            MetricCase(
                case_id=query.case_id,
                relevance=query.relevance,
                ranking=tuple(rankings[query.case_id]),
                slices=tuple(sorted({*query.slices, query.query_language})),
                forbidden=frozenset(query.forbidden),
                allowed_result_keys=frozenset(
                    record.memory_key
                    for record in corpus
                    if record.status == "active"
                    and (record.scope == "global" or record.project_id == query.project_id)
                    and (query.memory_type is None or record.memory_type == query.memory_type)
                ),
            )
            for query in queries
        ),
        known_keys=frozenset(record.memory_key for record in corpus),
        cutoff=5,
    )
    return {
        "aggregate": asdict(summary.aggregate),
        "slices": {name: asdict(value) for name, value in summary.slices.items()},
    }


async def _trial(
    repository: MemoryRepository,
    corpus: tuple[CorpusRecord, ...],
    queries: tuple[QueryRecord, ...],
    vectors: dict[str, list[float]],
    *,
    pool_size: int,
    rrf_k: int,
    repeats: int = 3,
) -> dict[str, Any]:
    key_by_id = {record.id: record.memory_key for record in corpus}
    rankings: dict[str, list[str]] = {}
    samples: list[float] = []
    stable = True
    returned = 0
    for query in queries:
        case_rankings: list[list[str]] = []
        for _ in range(repeats):
            started = perf_counter()
            dense, lexical = await asyncio.gather(
                repository.search_dense(
                    vectors[query.case_id],
                    query.project_id,
                    pool_size,
                    strategy="exact",
                    memory_type=query.memory_type,
                ),
                repository.search_lexical(
                    normalize_query_to_lexical_groups(query.query),
                    query.project_id,
                    pool_size,
                    memory_type=query.memory_type,
                ),
            )
            fused = reciprocal_rank_fusion(dense, lexical, k=rrf_k, limit=5)
            samples.append((perf_counter() - started) * 1000)
            case_rankings.append([key_by_id[result.id] for result in fused])
        stable &= all(ranking == case_rankings[0] for ranking in case_rankings)
        rankings[query.case_id] = case_rankings[0]
        returned += len(case_rankings[0])
    return {
        "config": {
            "dense_strategy": "exact",
            "fts_config": "simple",
            "pool": pool_size,
            "rrf_k": rrf_k,
        },
        "repeats": repeats,
        "rankings": rankings,
        "repeat_stable": stable,
        "results_returned": returned,
        "retrieval_fusion_p50_ms": median(samples),
        "retrieval_fusion_samples_ms": samples,
        "metrics": _metrics(corpus, queries, rankings),
        "context_tokens": None,
    }


async def run_experiments(output: Path, *, ann_noise_max: int = 10000) -> Path:
    if output.exists():
        raise FileExistsError("experiment output directory already exists")
    dataset = load_retrieval_split(RETRIEVAL_ROOT, "dev")
    base_settings = Settings().model_copy(update={"embedding_local_files_only": True})
    if cache_override := os.environ.get("ORNA_TEST_E5_CACHE_DIR"):
        base_settings = base_settings.model_copy(
            update={"embedding_cache_dir": Path(cache_override)}
        )
    repo_root = Path(__file__).resolve().parents[4]
    code_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], text=True)
    )
    measured_files = (
        *sorted(Path(__file__).parent.glob("*.py")),
        repo_root / "orna-memory-mcp/tests/unit/test_retrieval_experiments.py",
        repo_root / "orna-memory-mcp/tests/integration/test_retrieval_experiments.py",
    )
    file_hashes = {
        str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in measured_files
    }
    async with ephemeral_eval_database(base_settings) as db:
        embeddings = AsyncEmbeddingExecutor(
            EmbeddingService(db.settings),
            max_concurrency=db.settings.embedding_max_concurrency,
        )
        try:
            await load_eval_corpus(db.pool, dataset.corpus, embeddings, db.settings)
            vectors = {
                query.case_id: await embeddings.embed_query(query.query)
                for query in dataset.queries
            }
            database, corpus_info = await inspect_eval_database(db.pool)
            repository = MemoryRepository(db.pool, db.settings)
            tuning: list[dict[str, Any]] = []
            # Один фактор относительно 20/60, затем две заранее ограниченные комбинации.
            for pool_size, rrf_k in (
                (20, 60),
                (10, 60),
                (40, 60),
                (20, 20),
                (20, 100),
                (10, 20),
                (40, 100),
            ):
                tuning.append(
                    await _trial(
                        repository,
                        dataset.corpus,
                        dataset.queries,
                        vectors,
                        pool_size=pool_size,
                        rrf_k=rrf_k,
                    )
                )
            lexical = await run_lexical(db.pool, dataset.corpus, dataset.queries)
            trigram = await run_trigram(
                db.pool, dataset.corpus, dataset.queries, repository, vectors
            )
            noise_levels = tuple(level for level in (0, 1000, 10000) if level <= ann_noise_max)
            ann = await run_ann(
                db.pool,
                dataset.corpus,
                dataset.queries,
                vectors,
                noise_levels=noise_levels,
            )
        finally:
            await embeddings.aclose()
    artifact = {
        "schema_version": 1,
        "experiment": "p2-04",
        "split": "dev",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code": {"sha": code_sha, "dirty": dirty, "file_sha256": file_hashes},
        "dataset": {
            "version": dataset.manifest.dataset_version,
            "hashes": {name: item.sha256 for name, item in dataset.manifest.files.items()},
        },
        "config": {
            "baseline": "tests/retrieval/baseline.json",
            "embedding_profile": ACTIVE_EMBEDDING_PROFILE.version,
            "embedding_snapshot": ACTIVE_EMBEDDING_PROFILE.source_revision,
            "query_embeddings": "one real pinned E5 vector per dev query, reused across variants",
        },
        "database": database,
        "corpus_before_noise": corpus_info,
        "query_count": len(dataset.queries),
        "tuning": tuning,
        "lexical": lexical,
        "trigram": trigram,
        "ann": ann,
        "holdout_run": False,
        "context_tokens": None,
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "run.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated P2-04 dev experiments")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ann-noise-max", type=int, choices=(0, 1000, 10000), default=10000)
    args = parser.parse_args()
    print(asyncio.run(run_experiments(args.output, ann_noise_max=args.ann_noise_max)))


if __name__ == "__main__":
    main()
