"""Structured JSON и Markdown artifacts одного retrieval evaluation result."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from tests.evals.dataset import DatasetManifest
from tests.evals.metrics import MetricAggregate


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodeMetadata(_ReportModel):
    sha: str
    dirty: bool


class DatasetMetadata(_ReportModel):
    version: str
    hashes: dict[str, str]


class DatabaseMetadata(_ReportModel):
    postgresql_version: str
    pgvector_version: str


class EmbeddingMetadata(_ReportModel):
    profile: str
    model: str
    snapshot: str


class LexicalMetadata(_ReportModel):
    profile: str
    fts_config: str


class RuntimeMetadata(_ReportModel):
    python: str
    dependencies: dict[str, str]


class CorpusMetadata(_ReportModel):
    records: int
    index_bytes: int


class TokenMeasurement(_ReportModel):
    status: Literal["unavailable", "exact", "estimate"]
    value: int | None
    tokenizer: str | None
    tokenizer_version: str | None
    measured_scope: str


class LatencyMeasurement(_ReportModel):
    status: Literal["available", "unavailable"]
    samples: int
    p50_ms: float | None
    p95_ms: float | None


class QueryReport(_ReportModel):
    case_id: str
    slices: tuple[str, ...]
    ranking: tuple[str, ...]
    results_returned: int
    results_used: None = None
    retrieved_memory_tokens: TokenMeasurement
    serialized_context_tokens: TokenMeasurement
    latency: dict[str, LatencyMeasurement]


class ModeReport(_ReportModel):
    mode: str
    query_count: int
    results_returned: int
    aggregate_metrics: dict[str, float | int | None]
    slice_metrics: dict[str, dict[str, float | int | None]]
    latency: dict[str, LatencyMeasurement]
    queries: tuple[QueryReport, ...]


class EvaluationRunResult(_ReportModel):
    schema_version: Literal[1] = 1
    run_id: str
    created_at_utc: str
    split: str
    code: CodeMetadata
    dataset: DatasetMetadata
    config: dict[str, Any]
    seed: int
    database: DatabaseMetadata
    embedding: EmbeddingMetadata
    lexical: LexicalMetadata
    runtime: RuntimeMetadata
    hardware: dict[str, str | int | None]
    corpus: CorpusMetadata
    query_counts: dict[str, Any]
    warmup: int
    repeats: int
    modes: tuple[ModeReport, ...]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(ceil(quantile * len(ordered)) - 1, 0)]


def _latency(values: list[float | None]) -> LatencyMeasurement:
    available = [value for value in values if value is not None]
    if not available:
        return LatencyMeasurement(status="unavailable", samples=0, p50_ms=None, p95_ms=None)
    return LatencyMeasurement(
        status="available",
        samples=len(available),
        p50_ms=_percentile(available, 0.50),
        p95_ms=_percentile(available, 0.95),
    )


def _metric_values(aggregate: MetricAggregate, cutoff: int) -> dict[str, float | int | None]:
    return {
        "query_count": aggregate.query_count,
        "positive_queries": aggregate.positive_queries,
        "negative_queries": aggregate.negative_queries,
        f"recall@{cutoff}": aggregate.recall_at_k,
        f"mrr@{cutoff}": aggregate.mrr_at_k,
        f"ndcg@{cutoff}": aggregate.ndcg_at_k,
        f"false_positive@{cutoff}": aggregate.false_positive_at_k,
        "retrieval_abstention_accuracy": aggregate.abstention_accuracy,
        "top1_accuracy": aggregate.top1_accuracy,
    }


def _unavailable_tokens(scope: str) -> TokenMeasurement:
    return TokenMeasurement(
        status="unavailable",
        value=None,
        tokenizer=None,
        tokenizer_version=None,
        measured_scope=scope,
    )


def build_run_result(
    execution: Any,
    *,
    split: str,
    config: Any,
    manifest: DatasetManifest,
    code_sha: str,
    dirty: bool,
    database: dict[str, str],
    embedding: dict[str, str],
    lexical: dict[str, str],
    runtime: dict[str, Any],
    hardware: dict[str, str | int | None],
    corpus: dict[str, int],
    run_id: str,
    created_at_utc: str,
) -> EvaluationRunResult:
    """Строит единственный structured source для обоих artifact formats."""

    modes: list[ModeReport] = []
    for mode in execution.modes:
        query_reports = tuple(
            QueryReport(
                case_id=query.case_id,
                slices=query.slices,
                ranking=query.ranking,
                results_returned=query.results_returned,
                retrieved_memory_tokens=_unavailable_tokens(
                    "concatenated content of retrieved memories"
                ),
                serialized_context_tokens=_unavailable_tokens(
                    "serialized MCP-compatible returned memory context"
                ),
                latency={
                    "embedding": _latency(list(query.embedding_latency_ms)),
                    "retrieval_fusion": _latency(list(query.retrieval_fusion_latency_ms)),
                    "total_search": _latency(list(query.total_latency_ms)),
                },
            )
            for query in mode.queries
        )
        modes.append(
            ModeReport(
                mode=mode.mode,
                query_count=len(query_reports),
                results_returned=sum(query.results_returned for query in query_reports),
                aggregate_metrics=_metric_values(mode.metrics.aggregate, mode.metrics.cutoff),
                slice_metrics={
                    name: _metric_values(values, mode.metrics.cutoff)
                    for name, values in sorted(mode.metrics.slices.items())
                },
                latency={
                    "embedding": _latency(
                        [sample for query in mode.queries for sample in query.embedding_latency_ms]
                    ),
                    "retrieval_fusion": _latency(
                        [
                            sample
                            for query in mode.queries
                            for sample in query.retrieval_fusion_latency_ms
                        ]
                    ),
                    "total_search": _latency(
                        [sample for query in mode.queries for sample in query.total_latency_ms]
                    ),
                },
                queries=query_reports,
            )
        )

    reference_queries = execution.modes[0].queries
    slice_counts: dict[str, int] = {}
    for query in reference_queries:
        for slice_name in query.slices:
            slice_counts[slice_name] = slice_counts.get(slice_name, 0) + 1
    positive_count = sum(query.positive for query in reference_queries)
    return EvaluationRunResult(
        run_id=run_id,
        created_at_utc=created_at_utc,
        split=split,
        code=CodeMetadata(sha=code_sha, dirty=dirty),
        dataset=DatasetMetadata(
            version=manifest.dataset_version,
            hashes={name: record.sha256 for name, record in sorted(manifest.files.items())},
        ),
        config=config.model_dump(mode="json"),
        seed=config.seed,
        database=DatabaseMetadata.model_validate(database),
        embedding=EmbeddingMetadata.model_validate(embedding),
        lexical=LexicalMetadata.model_validate(lexical),
        runtime=RuntimeMetadata.model_validate(runtime),
        hardware=hardware,
        corpus=CorpusMetadata.model_validate(corpus),
        query_counts={
            "records": len(reference_queries),
            "positive": positive_count,
            "negative": len(reference_queries) - positive_count,
            "slices": dict(sorted(slice_counts.items())),
        },
        warmup=config.warmup,
        repeats=config.repeats,
        modes=tuple(modes),
    )


def render_markdown(result: EvaluationRunResult) -> str:
    """Рендерит human-readable summary без raw queries/content/credentials."""

    lines = [
        f"# Orna retrieval evaluation: {result.run_id}",
        "",
        f"- Split: `{result.split}`",
        f"- Code: `{result.code.sha}` (dirty: `{str(result.code.dirty).lower()}`)",
        f"- Dataset: `{result.dataset.version}`",
        f"- PostgreSQL / pgvector: `{result.database.postgresql_version}` / "
        f"`{result.database.pgvector_version}`",
        f"- Embedding profile: `{result.embedding.profile}`",
        f"- Warmup / repeats: `{result.warmup}` / `{result.repeats}`",
        "",
        "## Aggregate metrics",
        "",
        "| Mode | Recall@5 | MRR@5 | nDCG@5 | FP@5 | Abstention | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in result.modes:
        metrics = mode.aggregate_metrics
        total = mode.latency["total_search"]
        values = (
            metrics.get("recall@5"),
            metrics.get("mrr@5"),
            metrics.get("ndcg@5"),
            metrics.get("false_positive@5"),
            metrics.get("retrieval_abstention_accuracy"),
            total.p50_ms,
            total.p95_ms,
        )
        rendered = ["unavailable" if value is None else f"{value:.6g}" for value in values]
        lines.append(f"| {mode.mode} | " + " | ".join(rendered) + " |")

    lines.extend(["", "## Slice metrics", ""])
    for mode in result.modes:
        lines.extend(
            [
                f"### {mode.mode}",
                "",
                "| Slice | Positive | Negative | Recall@5 | Top-1 | FP@5 | Abstention |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, metrics in mode.slice_metrics.items():
            values = (
                metrics["positive_queries"],
                metrics["negative_queries"],
                metrics["recall@5"],
                metrics["top1_accuracy"],
                metrics["false_positive@5"],
                metrics["retrieval_abstention_accuracy"],
            )
            rendered = ["unavailable" if value is None else f"{value:.6g}" for value in values]
            lines.append(f"| {name} | " + " | ".join(rendered) + " |")
        lines.append("")

    lines.extend(["", "## Rankings", ""])
    for mode in result.modes:
        lines.append(f"### {mode.mode}")
        lines.append("")
        for query in mode.queries:
            ranking = ", ".join(f"`{key}`" for key in query.ranking) or "_(empty)_"
            lines.append(f"- `{query.case_id}`: {ranking}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_run_artifacts(
    result: EvaluationRunResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Атомарность форматов обеспечивается единым immutable result object."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "run.json"
    markdown_path = output_dir / "summary.md"
    json_text = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    json_path.write_text(json_text + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path
