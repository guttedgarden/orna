"""pg_trgm только по идентификаторам во временной eval DB."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from statistics import median
from time import perf_counter
from typing import Any

import asyncpg

from app.normalizer import normalize_query_to_lexical_groups, split_identifier
from app.repository import MemoryRepository
from tests.evals.dataset import CorpusRecord, QueryRecord
from tests.evals.metrics import MetricCase, evaluate_rankings

_IDENTIFIER_SHAPE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_./:\\-]*(?: [A-Za-z][A-Za-z0-9_./:\\-]*){0,2}$"
)


def _eligible_identifier_query(query: str) -> bool:
    """Синтаксический gate без доступа к labels или найденным candidates."""
    return bool(_IDENTIFIER_SHAPE.fullmatch(query.strip()))


def _fuse_three(
    dense: list[str],
    lexical: list[str],
    trigram: list[str],
    tie_break: dict[str, tuple[int, int]],
    *,
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for channel in (dense, lexical, trigram):
        for rank, key in enumerate(dict.fromkeys(channel), 1):
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
    return sorted(scores, key=lambda key: (-scores[key], best_rank[key], *tie_break[key]))[:5]


def _index_names(plan: dict[str, Any]) -> list[str]:
    names = [plan["Index Name"]] if "Index Name" in plan else []
    for child in plan.get("Plans", []):
        names.extend(_index_names(child))
    return names


async def prepare_trigram(pool: asyncpg.Pool, corpus: tuple[CorpusRecord, ...]) -> dict[str, Any]:
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        await conn.execute(
            """CREATE TABLE eval_identifier_signals (
                   memory_id uuid NOT NULL REFERENCES memories(id),
                   identifier text NOT NULL,
                   signal text NOT NULL,
                   PRIMARY KEY(memory_id,identifier)
               )"""
        )
        values = [
            (record.id, identifier, " ".join(split_identifier(identifier)).lower())
            for record in corpus
            for identifier in record.identifiers
        ]
        await conn.executemany("INSERT INTO eval_identifier_signals VALUES($1,$2,$3)", values)
        await conn.execute(
            "CREATE INDEX eval_identifier_trgm_gin ON eval_identifier_signals "
            "USING gin(signal gin_trgm_ops)"
        )
        await conn.execute("ANALYZE eval_identifier_signals")
        return {
            "extension_version": await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname='pg_trgm'"
            ),
            "signals": len(values),
            "index_bytes": int(
                await conn.fetchval("SELECT pg_relation_size('eval_identifier_trgm_gin'::regclass)")
            ),
        }


async def run_trigram(
    pool: asyncpg.Pool,
    corpus: tuple[CorpusRecord, ...],
    queries: tuple[QueryRecord, ...],
    repository: MemoryRepository,
    vectors: dict[str, list[float]],
    *,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
    candidate_pool: int = 10,
    repeats: int = 3,
) -> dict[str, Any]:
    info = await prepare_trigram(pool, corpus)
    key_by_id = {str(record.id): record.memory_key for record in corpus}
    tie_break = {record.memory_key: (record.logical_id.int, record.id.int) for record in corpus}
    base_channels: dict[str, tuple[list[str], list[str]]] = {}
    for query in queries:
        dense = await repository.search_dense(
            vectors[query.case_id],
            query.project_id,
            20,
            strategy="exact",
            memory_type=query.memory_type,
        )
        lexical = await repository.search_lexical(
            normalize_query_to_lexical_groups(query.query),
            query.project_id,
            20,
            memory_type=query.memory_type,
        )
        base_channels[query.case_id] = (
            [key_by_id[str(record.id)] for record, _ in dense],
            [key_by_id[str(record.id)] for record, _ in lexical],
        )
    trials: list[dict[str, Any]] = []
    sql = """
        SELECT id, identifier, signal_score FROM (
            SELECT DISTINCT ON (m.id) m.id, s.identifier,
                   similarity(s.signal,$2) AS signal_score
            FROM eval_identifier_signals AS s JOIN memories AS m ON m.id=s.memory_id
            WHERE s.signal % $2 AND m.status='active'
              AND (m.scope='global' OR m.project_id=$1)
              AND ($4::text IS NULL OR m.memory_type=$4)
            ORDER BY m.id, signal_score DESC
        ) AS candidates
        ORDER BY signal_score DESC, id ASC LIMIT $3
    """
    for threshold in thresholds:
        results: list[dict[str, Any]] = []
        for query in queries:
            samples: list[dict[str, Any]] = []
            async with pool.acquire() as conn, conn.transaction():
                await conn.execute(
                    "SELECT set_config('pg_trgm.similarity_threshold',$1,true)",
                    str(threshold),
                )
                plan_raw = await conn.fetchval(
                    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql,
                    query.project_id,
                    query.query.lower(),
                    candidate_pool,
                    query.memory_type,
                )
                plan = json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
                for _ in range(repeats):
                    started = perf_counter()
                    rows = await conn.fetch(
                        sql,
                        query.project_id,
                        query.query.lower(),
                        candidate_pool,
                        query.memory_type,
                    )
                    samples.append(
                        {
                            "latency_ms": (perf_counter() - started) * 1000,
                            "candidates": [
                                {
                                    "key": key_by_id[str(row["id"])],
                                    "identifier": row["identifier"],
                                    "similarity": float(row["signal_score"]),
                                }
                                for row in rows
                            ],
                        }
                    )
            dense_keys, lexical_keys = base_channels[query.case_id]
            eligible = _eligible_identifier_query(query.query)
            base_ranking = _fuse_three(dense_keys, lexical_keys, [], tie_break)
            fused_samples = [
                _fuse_three(
                    dense_keys,
                    lexical_keys,
                    [candidate["key"] for candidate in sample["candidates"]] if eligible else [],
                    tie_break,
                )
                for sample in samples
            ]
            results.append(
                {
                    "case_id": query.case_id,
                    "positive": bool(query.relevance),
                    "relevant": list(query.relevance),
                    "eligible": eligible,
                    "plan": plan[0],
                    "samples": samples,
                    "baseline_ranking": base_ranking,
                    "fused_ranking": fused_samples[0],
                    "repeat_stable_candidates": all(
                        sample["candidates"] == samples[0]["candidates"] for sample in samples
                    ),
                    "repeat_stable_fusion": all(
                        ranking == fused_samples[0] for ranking in fused_samples
                    ),
                }
            )
        negatives = [result for result in results if not result["positive"]]
        typo = [result for result in results if result["case_id"] == "dev-executor-002"]
        result_by_id = {result["case_id"]: result for result in results}
        metrics = evaluate_rankings(
            tuple(
                MetricCase(
                    case_id=query.case_id,
                    relevance=query.relevance,
                    ranking=tuple(result_by_id[query.case_id]["fused_ranking"]),
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
        trials.append(
            {
                "threshold": threshold,
                "results": results,
                "negative_with_candidate": sum(
                    bool(result["samples"][0]["candidates"]) for result in negatives
                ),
                "negative_count": len(negatives),
                "eligible_negative_with_candidate": sum(
                    result["eligible"] and bool(result["samples"][0]["candidates"])
                    for result in negatives
                ),
                "changed_fusion_queries": [
                    result["case_id"]
                    for result in results
                    if result["fused_ranking"] != result["baseline_ranking"]
                ],
                "typo_relevant_candidate": bool(
                    typo
                    and any(
                        candidate["key"] in typo[0]["relevant"]
                        for candidate in typo[0]["samples"][0]["candidates"]
                    )
                ),
                "latency_p50_ms": median(
                    sample["latency_ms"] for result in results for sample in result["samples"]
                ),
                "fusion_metrics": {
                    "aggregate": asdict(metrics.aggregate),
                    "slices": {name: asdict(value) for name, value in metrics.slices.items()},
                },
            }
        )
    typo_query = next(query for query in queries if query.case_id == "dev-executor-002")
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT set_config('pg_trgm.similarity_threshold','0.5',true)")
        await conn.execute("SELECT set_config('enable_seqscan','off',true)")
        await conn.execute("SELECT set_config('enable_indexscan','off',true)")
        forced_raw = await conn.fetchval(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
            "SELECT memory_id FROM eval_identifier_signals WHERE signal % $1",
            typo_query.query.lower(),
        )
        forced_plan = json.loads(forced_raw) if isinstance(forced_raw, str) else forced_raw
    return {
        "signal": "lowercase original identifier plus split_identifier terms",
        "eligibility": (
            "ASCII identifier-like query of 1-3 terms; syntax only, no labels/candidates"
        ),
        "operator": "pg_trgm % / similarity",
        "candidate_pool": candidate_pool,
        "fusion": "three-channel RRF k=60, top-5; dense/lexical pool 20",
        "repeats": repeats,
        **info,
        "forced_index_diagnostic": {
            "scope": "signal-only GIN usability; full filtered query plans are in trials",
            "plan": forced_plan[0],
            "index_names": _index_names(forced_plan[0]["Plan"]),
        },
        "trials": trials,
    }
