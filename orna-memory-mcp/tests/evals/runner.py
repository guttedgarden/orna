"""CLI и orchestration воспроизводимого retrieval evaluation run."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE
from app.embeddings import (
    AsyncEmbeddingBackend,
    AsyncEmbeddingExecutor,
    EmbeddingService,
    ModelCacheMissingError,
)
from app.model_cache import is_model_cache_ready
from app.models import MemoryScope, MemoryStatus
from app.normalizer import normalize_query_to_lexical_groups
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService
from tests.evals.database import (
    ephemeral_eval_database,
    inspect_eval_database,
    load_eval_corpus,
)
from tests.evals.dataset import (
    CorpusRecord,
    QueryRecord,
    RetrievalDataset,
    load_retrieval_split,
)
from tests.evals.metrics import (
    EvaluationContractError,
    MetricCase,
    MetricSummary,
    evaluate_rankings,
)
from tests.evals.report import build_run_result, write_run_artifacts

EvaluationMode = Literal["dense", "lexical", "hybrid"]
SplitName = Literal["dev", "holdout"]
_MODE_ORDER: tuple[EvaluationMode, ...] = ("dense", "lexical", "hybrid")


class EvaluationConfig(BaseModel):
    """Versioned runner configuration, independent from production defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    seed: int = 0
    cutoff: Literal[5] = 5
    dense_strategy: Literal["exact", "hnsw"] = "exact"
    candidate_pool_size: int = Field(default=20, ge=5)
    rrf_k: int = Field(default=60, ge=1)
    fts_config: Literal["simple"] = "simple"
    embedding_profile: Literal["e5-v1"] = "e5-v1"
    warmup: int = Field(default=0, ge=0)
    repeats: int = Field(default=1, ge=1)


@dataclass(frozen=True, slots=True)
class QueryExecution:
    """Стабильный ranking и latency samples одного query."""

    case_id: str
    positive: bool
    slices: tuple[str, ...]
    ranking: tuple[str, ...]
    results_returned: int
    embedding_latency_ms: tuple[float | None, ...]
    retrieval_fusion_latency_ms: tuple[float, ...]
    total_latency_ms: tuple[float, ...]
    retrieved_memory_tokens: None = None
    serialized_context_tokens: None = None
    results_used: None = None


@dataclass(frozen=True, slots=True)
class ModeExecution:
    """Результаты и метрики одного retrieval mode."""

    mode: EvaluationMode
    queries: tuple[QueryExecution, ...]
    metrics: MetricSummary


@dataclass(frozen=True, slots=True)
class EvaluationExecution:
    """Детерминированная часть run, не включающая volatile provenance."""

    modes: tuple[ModeExecution, ...]


class _SearchRecord(Protocol):
    id: UUID
    status: MemoryStatus
    scope: MemoryScope
    project_id: str | None
    memory_type: str


class _SearchService(Protocol):
    async def search(
        self,
        search_query: MemorySearchQuery,
        project_id: str | None,
    ) -> Sequence[_SearchRecord]: ...


class _TimedEmbeddings:
    """Собирает query embedding latency без изменения embedding backend."""

    def __init__(self, backend: AsyncEmbeddingBackend) -> None:
        self._backend = backend
        self.query_latency_ms: float | None = None

    def reset_query_latency(self) -> None:
        self.query_latency_ms = None

    async def embed_query(self, query: str) -> list[float]:
        started = perf_counter()
        try:
            return await self._backend.embed_query(query)
        finally:
            self.query_latency_ms = (perf_counter() - started) * 1000

    async def embed_memory(self, content: str) -> list[float]:
        return await self._backend.embed_memory(content)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Читает strict JSON config без неявного дополнения production settings."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return EvaluationConfig.model_validate(payload)


def apply_evaluation_config(settings: Settings, config: EvaluationConfig) -> Settings:
    """Применяет versioned eval parameters, не меняя production configuration."""

    return settings.model_copy(
        update={
            "dense_retrieval_strategy": config.dense_strategy,
            "retrieval_candidate_pool_size": config.candidate_pool_size,
            "rrf_k": config.rrf_k,
            "embedding_profile_version": config.embedding_profile,
            "embedding_local_files_only": True,
        }
    )


def safe_failure_message(error: BaseException) -> str:
    """Возвращает диагностируемый exit message без exception payload с DSN/secrets."""

    if isinstance(error, ModelCacheMissingError):
        return "evaluation failed: pinned E5 offline cache is unavailable"
    return f"evaluation failed: {type(error).__name__}"


def normalize_modes(modes: str | Iterable[str]) -> tuple[EvaluationMode, ...]:
    """Возвращает уникальные modes в canonical порядке."""

    raw_modes = modes.split(",") if isinstance(modes, str) else list(modes)
    requested = {mode.strip() for mode in raw_modes if mode.strip()}
    unsupported = requested - set(_MODE_ORDER)
    if unsupported:
        raise ValueError(f"unsupported evaluation mode: {sorted(unsupported)[0]}")
    if not requested:
        raise ValueError("at least one evaluation mode is required")
    return tuple(mode for mode in _MODE_ORDER if mode in requested)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI parser; dev остаётся единственным default split."""

    parser = argparse.ArgumentParser(description="Run isolated Orna retrieval evaluation")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--modes", default="dense,lexical,hybrid")
    parser.add_argument("--output", type=Path, required=True)
    return parser


class EvaluationRunner:
    """Исполняет queries через production repository/service boundaries."""

    def __init__(
        self,
        *,
        corpus: tuple[CorpusRecord, ...],
        config: EvaluationConfig,
        repository: MemoryRepository,
        embeddings: AsyncEmbeddingBackend,
        search_service: _SearchService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._corpus = corpus
        self._config = config
        self._repository = repository
        self._timed_embeddings = _TimedEmbeddings(embeddings)
        if search_service is None:
            if settings is None:
                raise ValueError("settings are required when constructing MemorySearchService")
            search_service = MemorySearchService(repository, self._timed_embeddings, settings)
            self._hybrid_embedding_is_timed = True
        else:
            self._hybrid_embedding_is_timed = False
        self._search_service = search_service
        self._corpus_by_id = {record.id: record for record in corpus}
        self._known_keys = frozenset(record.memory_key for record in corpus)

    @staticmethod
    def queries_for_split(
        dataset: RetrievalDataset,
        split: SplitName = "dev",
    ) -> tuple[QueryRecord, ...]:
        if split == "dev":
            return dataset.dev
        if split == "holdout":
            return dataset.holdout
        raise ValueError(f"unsupported dataset split: {split}")

    def _allowed_result_keys(self, query: QueryRecord) -> frozenset[str]:
        return frozenset(
            record.memory_key
            for record in self._corpus
            if record.status == "active"
            and (record.scope == "global" or record.project_id == query.project_id)
            and (query.memory_type is None or record.memory_type == query.memory_type)
        )

    def _stable_keys(
        self,
        query: QueryRecord,
        results: Sequence[_SearchRecord],
    ) -> tuple[str, ...]:
        keys: list[str] = []
        for result in results[: self._config.cutoff]:
            fixture = self._corpus_by_id.get(result.id)
            if fixture is None:
                raise EvaluationContractError(
                    f"query {query.case_id} returned unknown physical id {result.id}"
                )
            if result.status is not MemoryStatus.ACTIVE or fixture.status != "active":
                raise EvaluationContractError(
                    f"query {query.case_id} returned non-active result {fixture.memory_key}"
                )
            visible = result.scope is MemoryScope.GLOBAL or result.project_id == query.project_id
            if not visible:
                raise EvaluationContractError(
                    f"query {query.case_id} returned foreign project/scope result"
                )
            if query.memory_type is not None and result.memory_type != query.memory_type:
                raise EvaluationContractError(
                    f"query {query.case_id} returned result with wrong memory_type"
                )
            if (
                result.scope.value != fixture.scope
                or result.project_id != fixture.project_id
                or result.memory_type != fixture.memory_type
            ):
                raise EvaluationContractError(
                    f"query {query.case_id} returned result metadata inconsistent with corpus"
                )
            keys.append(fixture.memory_key)
        return tuple(keys)

    async def _search_once(
        self,
        query: QueryRecord,
        mode: EvaluationMode,
    ) -> tuple[tuple[str, ...], float | None, float, float]:
        self._timed_embeddings.reset_query_latency()
        started = perf_counter()
        if mode == "dense":
            query_embedding = await self._timed_embeddings.embed_query(query.query)
            retrieval_started = perf_counter()
            dense = await self._repository.search_dense(
                query_embedding,
                query.project_id,
                self._config.candidate_pool_size,
                self._config.dense_strategy,
                memory_type=query.memory_type,
            )
            results: Sequence[_SearchRecord] = [record for record, _distance in dense]
            retrieval_ms = (perf_counter() - retrieval_started) * 1000
        elif mode == "lexical":
            retrieval_started = perf_counter()
            lexical = await self._repository.search_lexical(
                normalize_query_to_lexical_groups(query.query),
                query.project_id,
                self._config.candidate_pool_size,
                memory_type=query.memory_type,
            )
            results = [record for record, _score in lexical]
            retrieval_ms = (perf_counter() - retrieval_started) * 1000
        else:
            results = await self._search_service.search(
                MemorySearchQuery(
                    query=query.query,
                    memory_type=query.memory_type,
                    limit=self._config.cutoff,
                ),
                project_id=query.project_id,
            )
            total_so_far_ms = (perf_counter() - started) * 1000
            embedding_ms = (
                self._timed_embeddings.query_latency_ms if self._hybrid_embedding_is_timed else None
            )
            retrieval_ms = max(total_so_far_ms - (embedding_ms or 0.0), 0.0)

        total_ms = (perf_counter() - started) * 1000
        return (
            self._stable_keys(query, results),
            self._timed_embeddings.query_latency_ms
            if mode != "hybrid" or self._hybrid_embedding_is_timed
            else None,
            retrieval_ms,
            total_ms,
        )

    async def _run_query(self, query: QueryRecord, mode: EvaluationMode) -> QueryExecution:
        for _ in range(self._config.warmup):
            await self._search_once(query, mode)

        rankings: list[tuple[str, ...]] = []
        embedding_samples: list[float | None] = []
        retrieval_samples: list[float] = []
        total_samples: list[float] = []
        for _ in range(self._config.repeats):
            ranking, embedding_ms, retrieval_ms, total_ms = await self._search_once(query, mode)
            rankings.append(ranking)
            embedding_samples.append(embedding_ms)
            retrieval_samples.append(retrieval_ms)
            total_samples.append(total_ms)

        if any(ranking != rankings[0] for ranking in rankings[1:]):
            raise EvaluationContractError(
                f"query {query.case_id} produced non-deterministic {mode} rankings"
            )

        slices = tuple(sorted({*query.slices, query.query_language}))
        return QueryExecution(
            case_id=query.case_id,
            positive=bool(query.relevance),
            slices=slices,
            ranking=rankings[0],
            results_returned=len(rankings[0]),
            embedding_latency_ms=tuple(embedding_samples),
            retrieval_fusion_latency_ms=tuple(retrieval_samples),
            total_latency_ms=tuple(total_samples),
        )

    async def run(
        self,
        queries: tuple[QueryRecord, ...],
        *,
        modes: Iterable[str],
    ) -> EvaluationExecution:
        """Выполняет modes последовательно и считает metrics из stable keys."""

        normalized_modes = normalize_modes(modes)
        mode_executions: list[ModeExecution] = []
        query_by_id = {query.case_id: query for query in queries}
        for mode in normalized_modes:
            query_results = tuple([await self._run_query(query, mode) for query in queries])
            metric_cases = tuple(
                MetricCase(
                    case_id=result.case_id,
                    relevance=query_by_id[result.case_id].relevance,
                    ranking=result.ranking,
                    slices=result.slices,
                    forbidden=frozenset(query_by_id[result.case_id].forbidden),
                    allowed_result_keys=self._allowed_result_keys(query_by_id[result.case_id]),
                )
                for result in query_results
            )
            metrics = evaluate_rankings(
                metric_cases,
                known_keys=self._known_keys,
                cutoff=self._config.cutoff,
            )
            mode_executions.append(ModeExecution(mode=mode, queries=query_results, metrics=metrics))
        return EvaluationExecution(modes=tuple(mode_executions))


def _git_metadata(repository_root: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status)


def _runtime_metadata() -> dict[str, Any]:
    distributions = (
        "asyncpg",
        "fastembed",
        "huggingface-hub",
        "mcp",
        "pgvector",
        "pydantic",
        "pydantic-settings",
        "uuid6",
    )
    return {
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in sorted(distributions)},
    }


def _hardware_metadata() -> dict[str, str | int | None]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }


async def run_isolated_evaluation(args: argparse.Namespace) -> tuple[Path, Path]:
    """Выполняет один isolated run с real pinned E5 и пишет два artifacts."""

    config = load_evaluation_config(args.config)
    modes = normalize_modes(args.modes)
    dataset = load_retrieval_split(args.config.parent, args.split)
    queries = dataset.queries

    base_settings = apply_evaluation_config(Settings(), config)
    cache_override = os.environ.get("ORNA_TEST_E5_CACHE_DIR")
    if cache_override:
        base_settings = base_settings.model_copy(
            update={"embedding_cache_dir": Path(cache_override)}
        )
    if not is_model_cache_ready(base_settings.embedding_cache_dir):
        raise ModelCacheMissingError("pinned E5 offline cache is unavailable")

    repository_root = Path(__file__).resolve().parents[3]
    code_sha, dirty = _git_metadata(repository_root)
    async with ephemeral_eval_database(base_settings) as database:
        eval_settings = apply_evaluation_config(database.settings, config)
        embeddings = AsyncEmbeddingExecutor(
            EmbeddingService(eval_settings),
            max_concurrency=eval_settings.embedding_max_concurrency,
        )
        try:
            await load_eval_corpus(database.pool, dataset.corpus, embeddings, eval_settings)
            repository = MemoryRepository(database.pool, eval_settings)
            runner = EvaluationRunner(
                corpus=dataset.corpus,
                config=config,
                repository=repository,
                embeddings=embeddings,
                settings=eval_settings,
            )
            execution = await runner.run(queries, modes=modes)
            database_metadata, corpus_metadata = await inspect_eval_database(database.pool)
        finally:
            await embeddings.aclose()

    profile = ACTIVE_EMBEDDING_PROFILE
    result = build_run_result(
        execution,
        split=args.split,
        config=config,
        manifest=dataset.manifest,
        code_sha=code_sha,
        dirty=dirty,
        database=database_metadata,
        embedding={
            "profile": profile.version,
            "model": profile.model_name,
            "snapshot": profile.source_revision,
        },
        lexical={
            "profile": eval_settings.lexical_profile_version,
            "fts_config": config.fts_config,
        },
        runtime=_runtime_metadata(),
        hardware=_hardware_metadata(),
        corpus=corpus_metadata,
        run_id=uuid4().hex,
        created_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    return write_run_artifacts(result, args.output)


def main() -> None:
    args = build_argument_parser().parse_args()
    try:
        json_path, markdown_path = asyncio.run(run_isolated_evaluation(args))
    except BaseException as error:
        print(safe_failure_message(error), file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Evaluation artifacts: {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
