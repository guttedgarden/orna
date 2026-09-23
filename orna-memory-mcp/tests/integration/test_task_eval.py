"""Pilot corpus search over a separate migrated PostgreSQL database."""

from pathlib import Path

from app.config import Settings
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService
from tests.evals.database import ephemeral_eval_database, load_eval_corpus
from tests.evals.task_runner import _load_corpus
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
