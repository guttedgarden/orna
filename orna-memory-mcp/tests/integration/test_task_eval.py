"""Pilot corpus search over a separate migrated PostgreSQL database."""

import json
from pathlib import Path

from app.config import Settings
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService
from tests.evals.database import ephemeral_eval_database, load_eval_corpus
from tests.evals.task_adapter import run_trial
from tests.evals.task_runner import _load_corpus, load_cases
from tests.integration.test_eval_runner import DeterministicEvalEmbeddings

TASK_ROOT = Path(__file__).parents[1] / "evals" / "tasks"


async def test_pilot_corpus_is_visible_only_in_isolated_project_and_is_read_only() -> None:
    corpus = _load_corpus(TASK_ROOT / "corpus.jsonl")
    embeddings = DeterministicEvalEmbeddings()
    base = Settings()
    async with ephemeral_eval_database(base) as database:
        settings = database.settings.model_copy(
            update={
                "dense_retrieval_strategy": "exact",
                "retrieval_candidate_pool_size": 20,
                "rrf_k": 60,
            }
        )
        await load_eval_corpus(database.pool, corpus, embeddings, settings)
        service = MemorySearchService(
            MemoryRepository(database.pool, settings), embeddings, settings
        )
        async with database.pool.acquire() as connection:
            before = await connection.fetchval("SELECT count(*) FROM memories")
        current = await service.search(
            MemorySearchQuery(query="AsyncEmbeddingExecutor cancellation shield drain"),
            "orna-pilot",
        )
        foreign = await service.search(
            MemorySearchQuery(query="AsyncEmbeddingExecutor cancellation shield drain"),
            "other-project",
        )
        async with database.pool.acquire() as connection:
            after = await connection.fetchval("SELECT count(*) FROM memories")
        assert before == after == len(corpus)
        assert any("Cancellation-safe" in record.content for record in current)
        assert foreign == []


async def test_trial_records_each_real_postgresql_search_latency(tmp_path: Path) -> None:
    corpus = _load_corpus(TASK_ROOT / "corpus.jsonl")
    case = load_cases(TASK_ROOT / "cases.jsonl")[0]
    source = tmp_path / case.rubric.source_path
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")

    class Completion:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
            self.calls += 1
            if self.calls == 1:
                message = {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "search-1",
                            "function": {
                                "name": "memory_search",
                                "arguments": json.dumps(
                                    {"query": "AsyncEmbeddingExecutor cancellation"}
                                ),
                            },
                        }
                    ],
                }
            else:
                message = {"content": "{}"}
            return {"choices": [{"message": message}], "usage": {"prompt_tokens": 3}}

    async with ephemeral_eval_database(Settings()) as database:
        embeddings = DeterministicEvalEmbeddings()
        await load_eval_corpus(database.pool, corpus, embeddings, database.settings)
        service = MemorySearchService(
            MemoryRepository(database.pool, database.settings), embeddings, database.settings
        )

        async def search(query: str) -> list[dict]:
            records = await service.search(MemorySearchQuery(query=query, limit=5), "orna-pilot")
            return [{"id": str(record.id), "content": record.content} for record in records]

        outcome = await run_trial(
            case=case,
            checkout=tmp_path,
            condition="on",
            completion=Completion(),
            search=search,
            prompts={"base": "same", "on_extra": "search", "after_tool": "continue"},
            allowed_paths={case.rubric.source_path},
            max_tool_rounds=2,
        )
    assert outcome["memory_search_calls"] == 1
    assert outcome["tool_events"][0]["results"]
    assert outcome["tool_events"][0]["search_latency_ms"] > 0
    assert outcome["memory_search_latency_ms"] == outcome["tool_events"][0]["search_latency_ms"]
    assert outcome["completion_events"][0]["output_tokens"] is None
    assert outcome["post_template_prompt_tokens_exact"] is None
