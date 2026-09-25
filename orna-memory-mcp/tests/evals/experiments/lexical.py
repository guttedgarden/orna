"""Eval-only FTS ablation с одинаковой document/query конфигурацией."""

from __future__ import annotations

from dataclasses import asdict
from statistics import median
from time import perf_counter
from typing import Any

import asyncpg

from app.normalizer import build_lexical_source, normalize_query_to_lexical_groups
from tests.evals.dataset import CorpusRecord, QueryRecord
from tests.evals.metrics import MetricCase, evaluate_rankings

_LANES = ("simple", "russian", "dual")


async def prepare_lexical_lanes(pool: asyncpg.Pool, corpus: tuple[CorpusRecord, ...]) -> None:
    """Создаёт отдельные GIN indexes только внутри ephemeral eval DB."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE eval_lexical_lanes (
                memory_id uuid PRIMARY KEY REFERENCES memories(id),
                natural_source text NOT NULL,
                code_source text NOT NULL,
                natural_ru tsvector GENERATED ALWAYS AS
                    (to_tsvector('russian', natural_source)) STORED,
                code_simple tsvector GENERATED ALWAYS AS
                    (to_tsvector('simple', code_source)) STORED
            )
            """
        )
        await conn.executemany(
            """INSERT INTO eval_lexical_lanes(memory_id,natural_source,code_source)
               VALUES($1,$2,$3)""",
            [
                (
                    record.id,
                    build_lexical_source(record.content, record.tags, []),
                    build_lexical_source("", [], record.identifiers),
                )
                for record in corpus
            ],
        )
        await conn.execute(
            "CREATE INDEX eval_natural_ru_gin ON eval_lexical_lanes USING gin(natural_ru)"
        )
        await conn.execute(
            "CREATE INDEX eval_code_simple_gin ON eval_lexical_lanes USING gin(code_simple)"
        )
        await conn.execute(
            "CREATE INDEX eval_russian_source_gin ON memories "
            "USING gin(to_tsvector('russian', lexical_source))"
        )
        await conn.execute("ANALYZE memories")
        await conn.execute("ANALYZE eval_lexical_lanes")


def _query_sql(lane: str, groups: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Сохраняет raw OR expanded внутри группы, AND между concept groups."""
    values: list[str] = []
    expressions: list[str] = []
    config = "russian" if lane == "russian" else "simple"
    for raw, expanded in groups:
        raw_param = len(values) + 2
        raw_sql = f"plainto_tsquery('{config}', ${raw_param})"
        values.append(raw)
        if raw != expanded:
            expanded_param = len(values) + 2
            expanded_sql = f"plainto_tsquery('{config}', ${expanded_param})"
            values.append(expanded)
            expressions.append(f"({raw_sql} || {expanded_sql})")
        else:
            expressions.append(raw_sql)
    simple_query = " && ".join(expressions)
    if lane == "simple":
        target = "m.lexical_text"
        join = ""
        query_sql = simple_query
        score = f"ts_rank_cd({target}, q.value)"
        predicate = f"{target} @@ q.value"
    elif lane == "russian":
        target = "to_tsvector('russian', m.lexical_source)"
        join = ""
        query_sql = simple_query
        score = f"ts_rank_cd({target}, q.value)"
        predicate = f"{target} @@ q.value"
    else:
        # В dual natural использует russian, code — simple; две независимые
        # tsquery строятся из тех же lexical concept groups.
        ru_values: list[str] = []
        ru_expressions: list[str] = []
        for raw, expanded in groups:
            raw_param = len(values) + len(ru_values) + 2
            raw_sql = f"plainto_tsquery('russian', ${raw_param})"
            ru_values.append(raw)
            if raw != expanded:
                expanded_param = len(values) + len(ru_values) + 2
                expanded_sql = f"plainto_tsquery('russian', ${expanded_param})"
                ru_values.append(expanded)
                ru_expressions.append(f"({raw_sql} || {expanded_sql})")
            else:
                ru_expressions.append(raw_sql)
        values += ru_values
        query_sql = f"{simple_query} AS code, {' && '.join(ru_expressions)} AS natural"
        join = "JOIN eval_lexical_lanes AS l ON l.memory_id=m.id"
        score = "(ts_rank_cd(l.code_simple,q.code)+ts_rank_cd(l.natural_ru,q.natural))"
        predicate = "(l.code_simple @@ q.code OR l.natural_ru @@ q.natural)"
    if lane != "dual":
        query_sql += " AS value"
    limit_param = len(values) + 2
    type_param = limit_param + 1
    sql = f"""
        WITH q AS (SELECT {query_sql})
        SELECT m.id, {score} AS lexical_score
        FROM memories AS m {join} CROSS JOIN q
        WHERE {predicate} AND m.status='active'
          AND (m.scope='global' OR m.project_id=$1)
          AND (${type_param}::text IS NULL OR m.memory_type=${type_param})
        ORDER BY lexical_score DESC, m.id ASC LIMIT ${limit_param}
    """
    return sql, values


async def run_lexical(
    pool: asyncpg.Pool,
    corpus: tuple[CorpusRecord, ...],
    queries: tuple[QueryRecord, ...],
    *,
    pool_size: int = 20,
    repeats: int = 3,
) -> dict[str, Any]:
    await prepare_lexical_lanes(pool, corpus)
    key_by_id = {str(record.id): record.memory_key for record in corpus}
    allowed = {
        query.case_id: frozenset(
            record.memory_key
            for record in corpus
            if record.status == "active"
            and (record.scope == "global" or record.project_id == query.project_id)
            and (query.memory_type is None or record.memory_type == query.memory_type)
        )
        for query in queries
    }
    lanes: dict[str, Any] = {}
    diagnostic_probes = (
        ("ru_morphology", "Агента", "git-user-commit"),
        ("identifier", "ResponseProviderExecutor", "response-provider-executor"),
        ("mixed", "Агента git commit", "git-user-commit"),
        ("negative", "несуществующий CockroachDB кластер", None),
    )
    for lane in _LANES:
        results: list[dict[str, Any]] = []
        for query in queries:
            groups = normalize_query_to_lexical_groups(query.query)
            sql, values = _query_sql(lane, groups)
            samples: list[dict[str, Any]] = []
            async with pool.acquire() as conn:
                for _ in range(repeats):
                    started = perf_counter()
                    rows = await conn.fetch(
                        sql, query.project_id, *values, pool_size, query.memory_type
                    )
                    samples.append(
                        {
                            "ranking": [key_by_id[str(row["id"])] for row in rows[:5]],
                            "latency_ms": (perf_counter() - started) * 1000,
                        }
                    )
            results.append(
                {
                    "case_id": query.case_id,
                    "slices": query.slices,
                    "samples": samples,
                    "ranking": samples[0]["ranking"],
                    "repeat_stable": all(s["ranking"] == samples[0]["ranking"] for s in samples),
                }
            )
        by_id = {result["case_id"]: result for result in results}
        metrics = evaluate_rankings(
            tuple(
                MetricCase(
                    case_id=query.case_id,
                    relevance=query.relevance,
                    ranking=tuple(by_id[query.case_id]["ranking"]),
                    slices=tuple(sorted({*query.slices, query.query_language})),
                    forbidden=frozenset(query.forbidden),
                    allowed_result_keys=allowed[query.case_id],
                )
                for query in queries
            ),
            known_keys=frozenset(key_by_id.values()),
            cutoff=5,
        )
        latencies = [s["latency_ms"] for r in results for s in r["samples"]]
        probes: list[dict[str, Any]] = []
        async with pool.acquire() as conn:
            for name, text, target in diagnostic_probes:
                probe_sql, probe_values = _query_sql(lane, normalize_query_to_lexical_groups(text))
                rows = await conn.fetch(probe_sql, "eval-a", *probe_values, pool_size, None)
                probes.append(
                    {
                        "name": name,
                        "target": target,
                        "ranking": [key_by_id[str(row["id"])] for row in rows[:5]],
                    }
                )
        lanes[lane] = {
            "metrics": {
                "aggregate": asdict(metrics.aggregate),
                "slices": {name: asdict(value) for name, value in metrics.slices.items()},
            },
            "latency_p50_ms": median(latencies),
            "queries": results,
            "diagnostic_probes_not_in_frozen_labels": probes,
        }
    async with pool.acquire() as conn:
        sizes = {
            name: int(await conn.fetchval("SELECT pg_relation_size($1::regclass)", name))
            for name in (
                "idx_memories_lexical_text",
                "eval_russian_source_gin",
                "eval_natural_ru_gin",
                "eval_code_simple_gin",
            )
        }
    return {"pool_size": pool_size, "repeats": repeats, "lanes": lanes, "index_bytes": sizes}
