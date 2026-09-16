"""Evaluation runner поверх реального PostgreSQL с deterministic embeddings."""

import hashlib
from pathlib import Path

import asyncpg

from app.config import Settings
from app.models import EMBEDDING_DIMENSION
from app.repository import MemoryRepository
from tests.evals.database import (
    EVAL_DATABASE_PREFIX,
    ephemeral_eval_database,
    inspect_eval_database,
    load_eval_corpus,
)
from tests.evals.dataset import QueryRecord, load_retrieval_dataset
from tests.evals.runner import EvaluationRunner, load_evaluation_config

RETRIEVAL_ROOT = Path(__file__).parents[1] / "retrieval"


class DeterministicEvalEmbeddings:
    """Fast test vectors; SQL, filtering, normalization и fusion остаются реальными."""

    @staticmethod
    def _vector(value: str) -> list[float]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % EMBEDDING_DIMENSION
        vector = [0.0] * EMBEDDING_DIMENSION
        vector[index] = 1.0
        return vector

    async def embed_memory(self, content: str) -> list[float]:
        return self._vector(content)

    async def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


async def test_runner_uses_migrated_ephemeral_database_and_is_reproducible() -> None:
    dataset = load_retrieval_dataset(RETRIEVAL_ROOT)
    selected_keys = {
        "storage-qdrant-old",
        "storage-postgres-current",
        "visibility-project-a",
        "visibility-project-b",
        "type-filter-decision",
        "type-filter-incident",
    }
    corpus = tuple(record for record in dataset.corpus if record.memory_key in selected_keys)
    queries = (
        QueryRecord(
            case_id="integration-positive",
            split_group="integration-positive",
            query="PostgreSQL exact retrieval baseline",
            project_id="eval-a",
            memory_type="decision",
            query_language="en",
            target_language="en",
            slices=["exact_identifier"],
            relevance={"storage-postgres-current": 2, "type-filter-decision": 1},
            forbidden=[
                "storage-qdrant-old",
                "visibility-project-b",
                "type-filter-incident",
            ],
        ),
        QueryRecord(
            case_id="integration-negative",
            split_group="integration-negative",
            query="Which Redis security key is active?",
            project_id="eval-a",
            memory_type="security",
            query_language="en",
            target_language="en",
            slices=["negative"],
            relevance={},
            forbidden=["visibility-project-b", "storage-qdrant-old"],
        ),
    )
    config = load_evaluation_config(RETRIEVAL_ROOT / "baseline.json").model_copy(
        update={"repeats": 2}
    )
    base_settings = Settings()
    database_name = ""

    async with ephemeral_eval_database(base_settings) as database:
        database_name = database.name
        assert database_name.startswith(EVAL_DATABASE_PREFIX)
        assert database.settings.postgres_db == database_name
        assert database.settings.database_url != base_settings.database_url

        eval_settings = database.settings.model_copy(
            update={
                "dense_retrieval_strategy": config.dense_strategy,
                "retrieval_candidate_pool_size": config.candidate_pool_size,
                "rrf_k": config.rrf_k,
            }
        )
        embeddings = DeterministicEvalEmbeddings()
        await load_eval_corpus(database.pool, corpus, embeddings, eval_settings)
        repository = MemoryRepository(database.pool, eval_settings)
        runner = EvaluationRunner(
            corpus=corpus,
            config=config,
            repository=repository,
            embeddings=embeddings,
            settings=eval_settings,
        )
        first = await runner.run(queries, modes=("dense", "lexical", "hybrid"))
        second = await runner.run(queries, modes=("dense", "lexical", "hybrid"))
        database_metadata, corpus_metadata = await inspect_eval_database(database.pool)

        assert tuple(mode.mode for mode in first.modes) == ("dense", "lexical", "hybrid")
        assert all(len(mode.queries) == len(queries) for mode in first.modes)
        assert [[query.ranking for query in mode.queries] for mode in first.modes] == [
            [query.ranking for query in mode.queries] for mode in second.modes
        ]
        assert [mode.metrics for mode in first.modes] == [mode.metrics for mode in second.modes]
        assert database_metadata["postgresql_version"]
        assert database_metadata["pgvector_version"]
        assert corpus_metadata["records"] == len(corpus)
        assert corpus_metadata["index_bytes"] > 0

    admin = await asyncpg.connect(
        host=base_settings.postgres_host,
        port=base_settings.postgres_port,
        user=base_settings.postgres_user,
        password=base_settings.postgres_password,
        database="template1",
    )
    try:
        still_exists = await admin.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1);",
            database_name,
        )
    finally:
        await admin.close()
    assert still_exists is False
