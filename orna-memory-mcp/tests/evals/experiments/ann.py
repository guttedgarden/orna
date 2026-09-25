"""Exact/HNSW experiment с tenant noise и фактическими query plans."""

from __future__ import annotations

import json
from statistics import median
from time import perf_counter
from typing import Any

import asyncpg

from tests.evals.dataset import CorpusRecord, QueryRecord

_SQL = """
SELECT *, embedding <=> $2 AS distance
FROM memories
WHERE status = 'active'
  AND (scope = 'global' OR project_id = $1)
  AND ($4::text IS NULL OR memory_type = $4)
ORDER BY distance ASC
LIMIT $3
"""


def _index_names(plan: dict[str, Any]) -> list[str]:
    names: list[str] = []
    if name := plan.get("Index Name"):
        names.append(name)
    for child in plan.get("Plans", []):
        names.extend(_index_names(child))
    return names


async def _measure(
    pool: asyncpg.Pool,
    query: QueryRecord,
    vector: list[float],
    *,
    strategy: str,
    ef_search: int,
    pool_size: int,
    repeats: int,
) -> dict[str, Any]:
    async with pool.acquire() as conn, conn.transaction():
        if strategy == "exact":
            await conn.execute("SELECT set_config('enable_indexscan', 'off', true)")
        else:
            await conn.execute("SELECT set_config('hnsw.ef_search', $1, true)", str(ef_search))
            await conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            if strategy == "forced_hnsw":
                await conn.execute("SELECT set_config('enable_seqscan', 'off', true)")
                await conn.execute("SELECT set_config('enable_bitmapscan', 'off', true)")
                await conn.execute("SELECT set_config('enable_sort', 'off', true)")
        raw_plan = await conn.fetchval(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + _SQL,
            query.project_id,
            vector,
            pool_size,
            query.memory_type,
        )
        plan = json.loads(raw_plan) if isinstance(raw_plan, str) else raw_plan
        samples: list[dict[str, Any]] = []
        for _ in range(repeats):
            started = perf_counter()
            rows = await conn.fetch(_SQL, query.project_id, vector, pool_size, query.memory_type)
            latency_ms = (perf_counter() - started) * 1000
            samples.append(
                {
                    "ids": [str(row["id"]) for row in rows],
                    "distances": [float(row["distance"]) for row in rows],
                    "latency_ms": latency_ms,
                }
            )
    return {
        "strategy": strategy,
        "ef_search": ef_search if strategy != "exact" else None,
        "plan": plan[0],
        "index_names": _index_names(plan[0]["Plan"]),
        "samples": samples,
        "latency_p50_ms": median(sample["latency_ms"] for sample in samples),
    }


async def _insert_noise(
    pool: asyncpg.Pool, count: int, offset: int, *, reverse: bool = False
) -> None:
    if count <= 0:
        return
    # Одинаковый passage vector создаёт контролируемый плотный cluster других tenants.
    # Шум не получает fixture key и никогда не меняет frozen corpus.
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memories (
                id, logical_id, revision, scope, project_id, memory_type, status,
                content, content_hash, tags, identifiers, lexical_source,
                lexical_profile_version, embedding, embedding_model,
                embedding_profile_version, provenance
            )
            SELECT md5('p204-noise-' || g::text)::uuid,
                   md5('p204-logical-' || g::text)::uuid,
                   1, 'project', 'foreign-noise-' || (g % 19)::text,
                   source.memory_type, 'active', 'synthetic foreign tenant noise',
                   decode(md5('p204-' || g::text) || md5('noise-' || g::text), 'hex'),
                   '{}'::text[], '{}'::text[], 'synthetic foreign tenant noise',
                   source.lexical_profile_version, source.embedding,
                   source.embedding_model, source.embedding_profile_version,
                   '{"source":{"kind":"eval_noise"}}'::jsonb
            FROM generate_series($1::int, $2::int, $3::int) AS g
            CROSS JOIN LATERAL (
                SELECT memory_type, lexical_profile_version, embedding,
                       embedding_model, embedding_profile_version
                FROM memories WHERE scope = 'global' AND status = 'active'
                ORDER BY id LIMIT 1
            ) AS source
            """,
            offset + count - 1 if reverse else offset,
            offset if reverse else offset + count - 1,
            -1 if reverse else 1,
        )


async def run_ann(
    pool: asyncpg.Pool,
    corpus: tuple[CorpusRecord, ...],
    queries: tuple[QueryRecord, ...],
    vectors: dict[str, list[float]],
    *,
    noise_levels: tuple[int, ...] = (0, 1000, 10000),
    repeats: int = 3,
    pool_size: int = 20,
) -> dict[str, Any]:
    """Сравнивает exact/planner/forced на одних vectors и WHERE."""
    selected = tuple(
        query
        for query in queries
        if query.case_id in {"dev-executor-001", "dev-network-001", "dev-negative-redis"}
    )
    key_by_id = {str(record.id): record.memory_key for record in corpus}
    levels: list[dict[str, Any]] = []
    previous = 0
    for level in noise_levels:
        await _insert_noise(pool, level - previous, previous)
        previous = level
        async with pool.acquire() as conn:
            counts = dict(
                await conn.fetchrow(
                    """
                    SELECT count(*)::int AS total,
                           count(*) FILTER (WHERE status='active')::int AS active,
                           count(*) FILTER (
                               WHERE status='active' AND project_id LIKE 'foreign-noise-%'
                           )::int AS foreign_noise
                    FROM memories
                    """
                )
            )
            await conn.execute("ANALYZE memories")
        cases: list[dict[str, Any]] = []
        for query in selected:
            vector = vectors[query.case_id]
            async with pool.acquire() as conn:
                visible = await conn.fetchval(
                    """SELECT count(*) FROM memories WHERE status='active'
                       AND (scope='global' OR project_id=$1)
                       AND ($2::text IS NULL OR memory_type=$2)""",
                    query.project_id,
                    query.memory_type,
                )
            exact = await _measure(
                pool,
                query,
                vector,
                strategy="exact",
                ef_search=40,
                pool_size=pool_size,
                repeats=repeats,
            )
            truth = exact["samples"][0]["ids"]
            cutoff_distance = exact["samples"][0]["distances"][-1]
            # Проверка ties среди всех видимых records, включая тех, кто пересёк LIMIT.
            async with pool.acquire() as conn:
                boundary = await conn.fetchrow(
                    """
                    SELECT count(*) FILTER (WHERE distance < $3)::int AS before,
                           count(*) FILTER (WHERE distance = $3)::int AS tied
                    FROM (
                        SELECT embedding <=> $2 AS distance FROM memories
                        WHERE status='active' AND (scope='global' OR project_id=$1)
                          AND ($4::text IS NULL OR memory_type=$4)
                    ) AS visible
                    """,
                    query.project_id,
                    vector,
                    cutoff_distance,
                    query.memory_type,
                )
            variants = [exact]
            for ef in (40, 100, 200):
                for strategy in ("planner", "forced_hnsw"):
                    result = await _measure(
                        pool,
                        query,
                        vector,
                        strategy=strategy,
                        ef_search=ef,
                        pool_size=pool_size,
                        repeats=repeats,
                    )
                    result["recall_vs_exact"] = [
                        len(set(sample["ids"]) & set(truth)) / len(truth)
                        for sample in result["samples"]
                    ]
                    result["actual_hnsw"] = "idx_memories_embedding_active" in result["index_names"]
                    variants.append(result)
            for variant in variants:
                for sample in variant["samples"]:
                    sample["keys"] = [
                        key_by_id.get(value, "<synthetic-noise>") for value in sample["ids"]
                    ]
            cases.append(
                {
                    "case_id": query.case_id,
                    "visible_count": int(visible),
                    "boundary": dict(boundary),
                    "variants": variants,
                }
            )
        levels.append({"noise": level, "counts": counts, "cases": cases})
    order_trials: list[dict[str, Any]] = []
    final_level = levels[-1]
    for order in ("ascending_noise", "descending_noise"):
        async with pool.acquire() as conn:
            if order == "ascending_noise":
                await conn.execute("REINDEX INDEX idx_memories_embedding_active")
            else:
                # Только synthetic rows в owned eval DB. Frozen corpus и IDs не меняются.
                await conn.execute("DROP INDEX idx_memories_embedding_active")
                await conn.execute("DELETE FROM memories WHERE project_id LIKE 'foreign-noise-%'")
                await conn.execute("VACUUM FULL memories")
        if order == "descending_noise":
            await _insert_noise(pool, noise_levels[-1], 0, reverse=True)
            async with pool.acquire() as conn:
                await conn.execute(
                    "CREATE INDEX idx_memories_embedding_active ON memories "
                    "USING hnsw (embedding vector_cosine_ops) WHERE status='active'"
                )
        async with pool.acquire() as conn:
            await conn.execute("ANALYZE memories")
        cases: list[dict[str, Any]] = []
        for baseline_case in final_level["cases"]:
            query = next(item for item in selected if item.case_id == baseline_case["case_id"])
            truth = set(baseline_case["variants"][0]["samples"][0]["ids"])
            variants: list[dict[str, Any]] = []
            for ef in (40, 100, 200):
                result = await _measure(
                    pool,
                    query,
                    vectors[query.case_id],
                    strategy="forced_hnsw",
                    ef_search=ef,
                    pool_size=pool_size,
                    repeats=repeats,
                )
                variants.append(
                    {
                        "ef_search": ef,
                        "index_names": result["index_names"],
                        "plan": result["plan"],
                        "latency_p50_ms": result["latency_p50_ms"],
                        "samples": result["samples"],
                        "recall_vs_exact": [
                            len(set(sample["ids"]) & truth) / len(truth)
                            for sample in result["samples"]
                        ],
                    }
                )
            cases.append({"case_id": query.case_id, "variants": variants})
        order_trials.append({"order": order, "noise": noise_levels[-1], "cases": cases})
    return {
        "pool_size": pool_size,
        "repeats": repeats,
        "levels": levels,
        "order_trials": order_trials,
    }
