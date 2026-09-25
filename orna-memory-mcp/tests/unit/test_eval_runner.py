"""Unit-контракт evaluation runner без PostgreSQL и model inference."""

from pathlib import Path
from shutil import copy2
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models import MemoryScope, MemoryStatus
from tests.evals.database import EvalDatabaseError, derive_eval_settings
from tests.evals.dataset import (
    CorpusRecord,
    QueryRecord,
    load_retrieval_dataset,
    load_retrieval_split,
)
from tests.evals.metrics import EvaluationContractError
from tests.evals.report import build_run_result, write_run_artifacts
from tests.evals.runner import (
    EvaluationConfig,
    EvaluationRunner,
    apply_evaluation_config,
    build_argument_parser,
    load_evaluation_config,
    load_run_config,
    normalize_modes,
    safe_failure_message,
)

RETRIEVAL_ROOT = Path(__file__).parents[1] / "retrieval"


def _corpus_record(
    *,
    memory_key: str,
    id_suffix: int,
    scope: str = "project",
    project_id: str | None = "eval-a",
    status: str = "active",
    memory_type: str = "decision",
) -> CorpusRecord:
    identifier = f"0199aabb-0000-7000-8000-{id_suffix:012d}"
    return CorpusRecord(
        memory_key=memory_key,
        id=identifier,
        logical_id=identifier,
        revision=1,
        supersedes_id=None,
        status=status,
        scope=scope,
        project_id=project_id,
        memory_type=memory_type,
        content=f"content for {memory_key}",
        tags=[],
        identifiers=[],
        source_ref="unit",
    )


def _query(*, relevance: dict[str, int] | None = None) -> QueryRecord:
    resolved_relevance = {"relevant": 2} if relevance is None else relevance
    return QueryRecord(
        case_id="dev-case",
        split_group="unit",
        query="PostgreSQL decision",
        project_id="eval-a",
        memory_type="decision",
        query_language="en",
        target_language="en",
        slices=["exact_identifier"] if resolved_relevance else ["negative"],
        relevance=resolved_relevance,
        forbidden=[],
    )


def _search_record(record: CorpusRecord) -> SimpleNamespace:
    return SimpleNamespace(
        id=record.id,
        status=MemoryStatus(record.status),
        scope=MemoryScope(record.scope),
        project_id=record.project_id,
        memory_type=record.memory_type,
        content=record.content,
    )


class _Embeddings:
    async def embed_query(self, query: str) -> list[float]:
        if query != "PostgreSQL decision":
            raise AssertionError("runner changed the query")
        return [1.0]

    async def embed_memory(self, _content: str) -> list[float]:
        raise AssertionError("runner must not embed corpus while executing queries")


class _Repository:
    def __init__(self, dense: list[Any], lexical: list[Any]) -> None:
        self.dense = dense
        self.lexical = lexical

    async def search_dense(
        self,
        query_embedding: list[float],
        project_id: str,
        limit: int,
        strategy: str,
        *,
        memory_type: str | None,
    ) -> list[tuple[Any, float]]:
        if (query_embedding, project_id, limit, strategy, memory_type) != (
            [1.0],
            "eval-a",
            20,
            "exact",
            "decision",
        ):
            raise AssertionError("dense mode did not preserve eval config and filters")
        return [(record, float(index)) for index, record in enumerate(self.dense, 1)]

    async def search_lexical(
        self,
        groups: list[tuple[str, str]],
        project_id: str,
        limit: int,
        *,
        memory_type: str | None,
    ) -> list[tuple[Any, float]]:
        if not groups or (project_id, limit, memory_type) != ("eval-a", 20, "decision"):
            raise AssertionError("lexical mode did not preserve query normalization and filters")
        return [(record, float(index)) for index, record in enumerate(self.lexical, 1)]


class _SearchService:
    def __init__(self, results: list[Any]) -> None:
        self.results = results

    async def search(self, search_query: Any, project_id: str) -> list[Any]:
        if (
            search_query.query,
            search_query.memory_type,
            search_query.limit,
            project_id,
        ) != ("PostgreSQL decision", "decision", 5, "eval-a"):
            raise AssertionError("hybrid mode did not use the application service contract")
        return self.results


def test_baseline_config_is_strict_and_matches_current_retrieval_profile(tmp_path: Path) -> None:
    config = load_evaluation_config(RETRIEVAL_ROOT / "baseline.json")

    assert config == EvaluationConfig(
        schema_version=1,
        seed=0,
        cutoff=5,
        dense_strategy="exact",
        candidate_pool_size=20,
        rrf_k=60,
        fts_config="simple",
        embedding_profile="e5-v1",
        warmup=0,
        repeats=1,
    )

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"schema_version":1,"unknown":true}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_evaluation_config(invalid_path)


def test_cli_defaults_to_dev_and_rejects_unknown_modes() -> None:
    args = build_argument_parser().parse_args(
        ["--config", str(RETRIEVAL_ROOT / "baseline.json"), "--output", "artifact"]
    )

    assert args.split == "dev"
    assert normalize_modes("hybrid,dense,lexical") == ("dense", "lexical", "hybrid")
    with pytest.raises(ValueError, match="unsupported"):
        normalize_modes("dense,unknown")


def test_final_run_repeats_rankings_without_changing_baseline_retrieval_config() -> None:
    args = build_argument_parser().parse_args(
        [
            "--config",
            str(RETRIEVAL_ROOT / "baseline.json"),
            "--output",
            "artifact",
            "--repeats",
            "3",
        ]
    )
    baseline = load_evaluation_config(args.config)
    final = load_run_config(args.config, args.repeats)

    assert baseline.repeats == 1
    assert final.repeats == 3
    assert final.model_copy(update={"repeats": 1}) == baseline
    with pytest.raises(ValidationError):
        load_run_config(args.config, 0)


def test_eval_config_overrides_retrieval_settings_without_exposing_runtime_dsn() -> None:
    base = Settings(
        _env_file=None,
        database_url="postgresql://user:secret@production.example/production",
        dense_retrieval_strategy="hnsw",
        retrieval_candidate_pool_size=99,
        rrf_k=7,
    )

    configured = apply_evaluation_config(base, EvaluationConfig())

    assert configured.dense_retrieval_strategy == "exact"
    assert configured.retrieval_candidate_pool_size == 20
    assert configured.rrf_k == 60
    assert configured.embedding_local_files_only is True
    assert "secret" not in safe_failure_message(
        RuntimeError("failed postgresql://user:secret@production.example/production")
    )


async def test_runner_uses_each_real_retrieval_boundary_and_stable_memory_keys() -> None:
    relevant = _corpus_record(memory_key="relevant", id_suffix=1)
    dense_only = _corpus_record(memory_key="dense-only", id_suffix=2)
    lexical_only = _corpus_record(memory_key="lexical-only", id_suffix=3)
    hybrid_only = _corpus_record(memory_key="hybrid-only", id_suffix=4)
    corpus = (relevant, dense_only, lexical_only, hybrid_only)
    runner = EvaluationRunner(
        corpus=corpus,
        config=load_evaluation_config(RETRIEVAL_ROOT / "baseline.json"),
        repository=_Repository(
            dense=[_search_record(relevant), _search_record(dense_only)],
            lexical=[_search_record(relevant), _search_record(lexical_only)],
        ),
        search_service=_SearchService([_search_record(relevant), _search_record(hybrid_only)]),
        embeddings=_Embeddings(),
    )

    execution = await runner.run((_query(),), modes=("hybrid", "lexical", "dense"))

    assert tuple(mode.mode for mode in execution.modes) == ("dense", "lexical", "hybrid")
    assert execution.modes[0].queries[0].ranking == ("relevant", "dense-only")
    assert execution.modes[1].queries[0].ranking == ("relevant", "lexical-only")
    assert execution.modes[2].queries[0].ranking == ("relevant", "hybrid-only")
    assert execution.modes[2].metrics.slices["en"].recall_at_k == 1.0
    assert execution.modes[2].metrics.slices["exact_identifier"].top1_accuracy == 1.0


async def test_runner_rejects_non_deterministic_repeat_rankings() -> None:
    relevant = _corpus_record(memory_key="relevant", id_suffix=1)
    distractor = _corpus_record(memory_key="distractor", id_suffix=2)

    class AlternatingRepository(_Repository):
        calls = 0

        async def search_dense(self, *args: Any, **kwargs: Any) -> list[tuple[Any, float]]:
            self.calls += 1
            records = self.dense if self.calls % 2 else list(reversed(self.dense))
            return [(record, float(index)) for index, record in enumerate(records, 1)]

    runner = EvaluationRunner(
        corpus=(relevant, distractor),
        config=EvaluationConfig(repeats=2),
        repository=AlternatingRepository(
            dense=[_search_record(relevant), _search_record(distractor)], lexical=[]
        ),
        search_service=_SearchService([]),
        embeddings=_Embeddings(),
    )

    with pytest.raises(EvaluationContractError, match="non-deterministic"):
        await runner.run((_query(),), modes=("dense",))


async def test_runner_rejects_result_whose_runtime_state_is_not_active() -> None:
    relevant = _corpus_record(memory_key="relevant", id_suffix=1)
    returned = _search_record(relevant)
    returned.status = MemoryStatus.SUPERSEDED
    runner = EvaluationRunner(
        corpus=(relevant,),
        config=EvaluationConfig(),
        repository=_Repository(dense=[returned], lexical=[]),
        search_service=_SearchService([]),
        embeddings=_Embeddings(),
    )

    with pytest.raises(EvaluationContractError, match="non-active"):
        await runner.run((_query(),), modes=("dense",))


async def test_negative_query_still_rejects_non_active_runtime_result() -> None:
    fixture = _corpus_record(memory_key="hard-negative", id_suffix=1)
    returned = _search_record(fixture)
    returned.status = MemoryStatus.SUPERSEDED
    runner = EvaluationRunner(
        corpus=(fixture,),
        config=EvaluationConfig(),
        repository=_Repository(dense=[returned], lexical=[]),
        search_service=_SearchService([]),
        embeddings=_Embeddings(),
    )

    with pytest.raises(EvaluationContractError, match="non-active"):
        await runner.run((_query(relevance={}),), modes=("dense",))


def test_p2_01_dataset_is_loaded_once_and_dev_is_selected_without_reading_holdout_results() -> None:
    dataset = load_retrieval_dataset(RETRIEVAL_ROOT)

    assert EvaluationRunner.queries_for_split(dataset, "dev") is dataset.dev
    assert EvaluationRunner.queries_for_split(dataset, "holdout") is dataset.holdout


def test_dev_split_loader_does_not_open_or_require_holdout_labels(tmp_path: Path) -> None:
    for file_name in ("dataset-manifest.json", "corpus.jsonl", "memory_dev.jsonl"):
        copy2(RETRIEVAL_ROOT / file_name, tmp_path / file_name)

    selected = load_retrieval_split(tmp_path, "dev")

    assert selected.name == "dev"
    assert len(selected.queries) == 24
    assert not (tmp_path / "memory_holdout.jsonl").exists()


def test_eval_settings_replace_any_configured_target_database() -> None:
    base = Settings(
        _env_file=None,
        postgres_host="127.0.0.1",
        postgres_port=5432,
        postgres_user="orna",
        postgres_password="unit-secret",
        postgres_db="production",
        database_url="postgresql://orna:unit-secret@127.0.0.1/production",
    )

    target = derive_eval_settings(base, "orna_eval_unit_123")

    assert target.postgres_db == "orna_eval_unit_123"
    assert target.database_url is not None
    assert target.database_url.endswith("/orna_eval_unit_123")
    assert "/production" not in target.database_url
    with pytest.raises(EvalDatabaseError, match="prefix"):
        derive_eval_settings(base, "production")


def test_eval_database_rejects_non_loopback_admin_host() -> None:
    remote = Settings(_env_file=None, postgres_host="production.example")

    with pytest.raises(EvalDatabaseError, match="loopback"):
        derive_eval_settings(remote, "orna_eval_unit_123")


async def test_json_and_markdown_are_rendered_from_one_structured_result(tmp_path: Path) -> None:
    relevant = _corpus_record(memory_key="relevant", id_suffix=1)
    runner = EvaluationRunner(
        corpus=(relevant,),
        config=EvaluationConfig(),
        repository=_Repository(dense=[_search_record(relevant)], lexical=[]),
        search_service=_SearchService([_search_record(relevant)]),
        embeddings=_Embeddings(),
    )
    negative = _query(relevance={}).model_copy(update={"case_id": "negative-case"})
    execution = await runner.run((_query(), negative), modes=("dense",))
    dataset = load_retrieval_dataset(RETRIEVAL_ROOT)
    result = build_run_result(
        execution,
        split="dev",
        config=EvaluationConfig(),
        manifest=dataset.manifest,
        code_sha="ca745130933498871d8d07c7f62571b46b1c165e",
        dirty=True,
        database={"postgresql_version": "16.4", "pgvector_version": "0.8.0"},
        embedding={
            "profile": "e5-v1",
            "model": "intfloat/multilingual-e5-large",
            "snapshot": "pinned-snapshot",
        },
        lexical={"profile": "lexical-v1", "fts_config": "simple"},
        runtime={"python": "3.12", "dependencies": {"asyncpg": "0.31.0"}},
        hardware={"machine": "test"},
        corpus={"records": 1, "index_bytes": 1024},
        run_id="unit-run",
        created_at_utc="2026-09-15T00:00:00Z",
    )

    json_path, markdown_path = write_run_artifacts(result, tmp_path / "artifact")
    json_text = json_path.read_text(encoding="utf-8")
    markdown_text = markdown_path.read_text(encoding="utf-8")

    assert result.modes[0].aggregate_metrics["recall@5"] == 1.0
    assert result.modes[0].queries[0].ranking == ("relevant",)
    assert result.modes[0].queries[0].results_used is None
    assert result.modes[0].queries[0].retrieved_memory_tokens.status == "unavailable"
    assert result.query_counts == {
        "records": 2,
        "positive": 1,
        "negative": 1,
        "slices": {"en": 2, "exact_identifier": 1, "negative": 1},
    }
    assert '"run_id": "unit-run"' in json_text
    assert "unit-run" in markdown_text
    assert "relevant" in markdown_text
    assert "## Slice metrics" in markdown_text
    assert "| exact_identifier | 1 | 0 | 1 | 1 | unavailable | unavailable |" in markdown_text
    assert "| negative | 0 | 1 | unavailable | unavailable | 1 | 0 |" in markdown_text
    assert "PostgreSQL decision" not in json_text
    assert "PostgreSQL decision" not in markdown_text
