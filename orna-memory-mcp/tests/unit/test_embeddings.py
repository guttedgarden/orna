import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import (
    AsyncEmbeddingExecutor,
    EmbeddingOutputError,
    EmbeddingService,
    ModelCacheMissingError,
    ensure_prefix,
)


class TestPrefixHandling:
    """Verify query: and passage: prefix enforcement."""

    def test_ensure_prefix_prepends_to_plain_text(self):
        result = ensure_prefix("How do we run migrations?", "query: ")
        assert result == "query: How do we run migrations?"

    def test_ensure_prefix_does_not_duplicate_existing_prefix(self):
        result = ensure_prefix("query: How do we run migrations?", "query: ")
        assert result == "query: How do we run migrations?"

    def test_ensure_prefix_strips_multiple_prefixes(self):
        result = ensure_prefix("query: query: How do we run migrations?", "query: ")
        assert result == "query: How do we run migrations?"

    def test_ensure_prefix_case_insensitive(self):
        result = ensure_prefix("QUERY: How do we run migrations?", "query: ")
        assert result == "query: How do we run migrations?"
        result_mixed = ensure_prefix("Query: How do we run migrations?", "query: ")
        assert result_mixed == "query: How do we run migrations?"

    def test_ensure_prefix_passage(self):
        result = ensure_prefix("Postgres 16 is used for storage", "passage: ")
        assert result == "passage: Postgres 16 is used for storage"

    def test_ensure_prefix_passage_does_not_duplicate(self):
        result = ensure_prefix("passage: Postgres 16 is used for storage", "passage: ")
        assert result == "passage: Postgres 16 is used for storage"

    def test_ensure_prefix_passage_multiple_stripped(self):
        result = ensure_prefix("passage: passage: Postgres 16", "passage: ")
        assert result == "passage: Postgres 16"

    def test_ensure_prefix_preserves_words_without_colon(self):
        # A query starting with the word 'query' as a subject, not a prefix tag
        result = ensure_prefix("Query performance is degraded", "query: ")
        assert result == "query: Query performance is degraded"

        result_passage = ensure_prefix("Passage through the network", "passage: ")
        assert result_passage == "passage: Passage through the network"


class TestEmbeddingService:
    """Verify EmbeddingService calls and output structure."""

    @pytest.fixture
    def mock_model(self):
        mock = MagicMock()
        # Return a mock 1024-dim numpy array
        fake_vector = np.full(1024, 0.05, dtype=np.float32)
        mock.embed.side_effect = lambda texts: (fake_vector for _ in texts)
        return mock

    @pytest.fixture
    def service(self) -> EmbeddingService:
        return EmbeddingService(Settings(_env_file=None))

    def test_embed_query_invokes_model_with_prefix(self, service, mock_model):
        with patch.object(service, "_model", mock_model):
            vec = service.embed_query("find database architecture")
            assert len(vec) == 1024
            assert isinstance(vec, list)
            assert isinstance(vec[0], float)
            mock_model.embed.assert_called_once_with(["query: find database architecture"])

    def test_embed_query_avoids_double_prefix(self, service, mock_model):
        with patch.object(service, "_model", mock_model):
            service.embed_query("query: find database architecture")
            mock_model.embed.assert_called_once_with(["query: find database architecture"])

    def test_embed_memory_invokes_model_with_passage_prefix(self, service, mock_model):
        with patch.object(service, "_model", mock_model):
            vec = service.embed_memory("Postgres 16 + pgvector was chosen in ADR-0001")
            assert len(vec) == 1024
            mock_model.embed.assert_called_once_with(
                ["passage: Postgres 16 + pgvector was chosen in ADR-0001"]
            )

    def test_embed_memories_batch(self, service, mock_model):
        with patch.object(service, "_model", mock_model):
            vectors = service.embed_memories(["text 1", "passage: text 2"])
            assert len(vectors) == 2
            assert len(vectors[0]) == 1024
            assert len(vectors[1]) == 1024
            mock_model.embed.assert_called_once_with(["passage: text 1", "passage: text 2"])

    def test_embed_memories_empty(self, service, mock_model):
        with patch.object(service, "_model", mock_model):
            vectors = service.embed_memories([])
            assert vectors == []
            mock_model.embed.assert_not_called()

    @pytest.mark.parametrize("invalid_contents", ["one memory", b"one memory"])
    def test_embed_memories_rejects_single_string_sequences(self, service, invalid_contents):
        with pytest.raises(TypeError, match="not a single string"):
            service.embed_memories(invalid_contents)

    def test_embed_memories_rejects_non_string_items(self, service):
        with pytest.raises(TypeError, match="every content item"):
            service.embed_memories(["valid", 42])

    @pytest.mark.parametrize(
        ("vector", "message"),
        [
            (np.full(1023, 0.05, dtype=np.float32), "dimension"),
            (np.full(1024, np.nan, dtype=np.float32), "finite"),
            (np.zeros(1024, dtype=np.float32), "non-zero norm"),
            ("not-a-vector", "one-dimensional"),
        ],
    )
    def test_rejects_invalid_model_output(self, service, mock_model, vector, message):
        mock_model.embed.return_value = iter([vector])
        mock_model.embed.side_effect = None

        with patch.object(service, "_model", mock_model):
            with pytest.raises(EmbeddingOutputError, match=message):
                service.embed_query("query")

    def test_rejects_missing_model_output(self, service, mock_model):
        mock_model.embed.return_value = iter([])
        mock_model.embed.side_effect = None

        with patch.object(service, "_model", mock_model):
            with pytest.raises(EmbeddingOutputError, match="unexpected number"):
                service.embed_query("query")

    def test_profile_dimension_matches_fastembed_catalog(self):
        service = EmbeddingService(Settings(_env_file=None))

        assert service.profile.dimension == 1024

    def test_rejects_fastembed_catalog_dimension_mismatch(self):
        with patch("app.embeddings.TextEmbedding.get_embedding_size", return_value=384):
            with pytest.raises(ValueError, match="dimension does not match"):
                EmbeddingService(Settings(_env_file=None))

    def test_offline_model_requires_pinned_snapshot(self, tmp_path):
        service = EmbeddingService(
            Settings(embedding_cache_dir=tmp_path, embedding_local_files_only=True, _env_file=None)
        )

        with pytest.raises(ModelCacheMissingError, match=r"python -m app\.model_cache"):
            _ = service.model

    def test_offline_model_uses_pinned_snapshot(self, tmp_path, mock_model):
        service = EmbeddingService(
            Settings(embedding_cache_dir=tmp_path, embedding_local_files_only=True, _env_file=None)
        )
        snapshot_path = service.profile.snapshot_path(tmp_path)
        snapshot_path.mkdir(parents=True)

        with patch("app.embeddings.TextEmbedding", return_value=mock_model) as constructor:
            assert service.model is mock_model

        constructor.assert_called_once_with(
            model_name="intfloat/multilingual-e5-large",
            cache_dir=str(tmp_path),
            threads=2,
            local_files_only=True,
            specific_model_path=str(snapshot_path),
        )

    def test_model_initializes_once_under_concurrency(self, mock_model):
        service = EmbeddingService(Settings(embedding_local_files_only=False, _env_file=None))

        def slow_constructor(**_kwargs):
            time.sleep(0.05)
            return mock_model

        with patch("app.embeddings.TextEmbedding", side_effect=slow_constructor) as constructor:
            with ThreadPoolExecutor(max_workers=8) as executor:
                models = list(executor.map(lambda _index: service.model, range(8)))

        assert all(model is mock_model for model in models)
        constructor.assert_called_once_with(
            model_name="intfloat/multilingual-e5-large",
            cache_dir=str(service.cache_dir),
            threads=2,
            local_files_only=False,
        )


class TestAsyncEmbeddingExecutor:
    async def test_cancelled_caller_keeps_slot_until_running_thread_finishes(self):
        first_started = Event()
        second_started = Event()
        release_first = Event()
        active = 0
        maximum_active = 0
        counter_lock = Lock()

        def embed_query(query: str) -> list[float]:
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                if query == "first":
                    first_started.set()
                    assert release_first.wait(timeout=1)
                else:
                    second_started.set()
                return [float(len(query))]
            finally:
                with counter_lock:
                    active -= 1

        service = MagicMock()
        service.embed_query.side_effect = embed_query
        executor = AsyncEmbeddingExecutor(service, max_concurrency=1)

        first_caller = asyncio.create_task(executor.embed_query("first"))
        assert await asyncio.to_thread(first_started.wait, 1)
        first_caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_caller

        second_caller = asyncio.create_task(executor.embed_query("second"))
        assert not await asyncio.to_thread(second_started.wait, 0.05)

        release_first.set()
        assert await second_caller == [6.0]
        assert maximum_active == 1

    async def test_aclose_drains_owned_work_and_rejects_new_submissions(self):
        started = Event()
        release = Event()

        def embed_query(_query: str) -> list[float]:
            started.set()
            assert release.wait(timeout=1)
            return [1.0]

        service = MagicMock()
        service.embed_query.side_effect = embed_query
        executor = AsyncEmbeddingExecutor(service, max_concurrency=1)

        running = asyncio.create_task(executor.embed_query("running"))
        assert await asyncio.to_thread(started.wait, 1)

        closing = asyncio.create_task(executor.aclose())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="closed"):
            await executor.embed_query("rejected")
        assert not closing.done()

        release.set()
        assert await running == [1.0]
        await closing

        with pytest.raises(RuntimeError, match="closed"):
            await executor.embed_query("still-rejected")

    async def test_repeated_cancellations_do_not_leak_or_double_release_slots(self):
        starts = {query: Event() for query in ("first", "second", "third", "fourth")}
        releases = {query: Event() for query in starts}
        active = 0
        maximum_active = 0
        counter_lock = Lock()

        def embed_query(query: str) -> list[float]:
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                starts[query].set()
                assert releases[query].wait(timeout=1)
                return [float(len(query))]
            finally:
                with counter_lock:
                    active -= 1

        service = MagicMock()
        service.embed_query.side_effect = embed_query
        executor = AsyncEmbeddingExecutor(service, max_concurrency=1)

        first = asyncio.create_task(executor.embed_query("first"))
        assert await asyncio.to_thread(starts["first"].wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        releases["first"].set()

        second = asyncio.create_task(executor.embed_query("second"))
        assert await asyncio.to_thread(starts["second"].wait, 1)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        releases["second"].set()

        third = asyncio.create_task(executor.embed_query("third"))
        fourth = asyncio.create_task(executor.embed_query("fourth"))
        assert await asyncio.to_thread(starts["third"].wait, 1)
        assert not await asyncio.to_thread(starts["fourth"].wait, 0.05)
        releases["third"].set()
        assert await third == [5.0]

        assert await asyncio.to_thread(starts["fourth"].wait, 1)
        releases["fourth"].set()
        assert await fourth == [6.0]
        assert maximum_active == 1

    async def test_abandoned_backend_exception_is_extracted_and_releases_slot(self):
        started = Event()
        release = Event()
        failed = Event()
        loop_errors: list[dict[str, object]] = []

        class BackendFailure(RuntimeError):
            pass

        def embed_query(query: str) -> list[float]:
            if query == "failing":
                started.set()
                assert release.wait(timeout=1)
                failed.set()
                raise BackendFailure("backend failed")
            return [1.0]

        service = MagicMock()
        service.embed_query.side_effect = embed_query
        executor = AsyncEmbeddingExecutor(service, max_concurrency=1)
        loop = asyncio.get_running_loop()
        previous_exception_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            caller = asyncio.create_task(executor.embed_query("failing"))
            assert await asyncio.to_thread(started.wait, 1)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller

            release.set()
            assert await asyncio.to_thread(failed.wait, 1)
            assert await executor.embed_query("recovery") == [1.0]
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_exception_handler)

        assert loop_errors == []

    async def test_running_thread_does_not_block_event_loop_heartbeat(self):
        started = Event()
        release = Event()
        heartbeat = asyncio.Event()

        def embed_query(_query: str) -> list[float]:
            started.set()
            assert release.wait(timeout=1)
            return [1.0]

        service = MagicMock()
        service.embed_query.side_effect = embed_query
        executor = AsyncEmbeddingExecutor(service, max_concurrency=1)

        running = asyncio.create_task(executor.embed_query("blocked"))
        assert await asyncio.to_thread(started.wait, 1)

        async def beat() -> None:
            await asyncio.sleep(0)
            heartbeat.set()

        beat_task = asyncio.create_task(beat())
        await asyncio.wait_for(heartbeat.wait(), timeout=0.2)
        await beat_task

        release.set()
        assert await running == [1.0]

    async def test_all_embedding_operations_share_threaded_submit_path(self):
        caller_thread = get_ident()

        class SynchronousService:
            def embed_query(self, _query: str) -> list[int]:
                return [get_ident()]

            def embed_memory(self, _content: str) -> list[int]:
                return [get_ident()]

            def embed_memories(self, _contents: list[str]) -> list[list[int]]:
                return [[get_ident()]]

        executor = AsyncEmbeddingExecutor(SynchronousService(), max_concurrency=1)

        assert (await executor.embed_query("query"))[0] != caller_thread
        assert (await executor.embed_memory("memory"))[0] != caller_thread
        assert (await executor.embed_memories(["one"]))[0][0] != caller_thread

    async def test_direct_backend_error_reaches_caller_and_does_not_lose_slot(self):
        class BackendFailure(RuntimeError):
            pass

        calls = 0

        def embed_query(_query: str) -> list[float]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise BackendFailure("backend failed")
            return [1.0]

        service = MagicMock()
        service.embed_query.side_effect = embed_query
        executor = AsyncEmbeddingExecutor(service, max_concurrency=1)

        with pytest.raises(BackendFailure, match="backend failed"):
            await executor.embed_query("fails")
        assert await executor.embed_query("recovers") == [1.0]

    async def test_bounds_concurrent_inference(self):
        active = 0
        maximum_active = 0
        two_started = Event()
        release = Event()
        counter_lock = Lock()

        def embed_query(query: str) -> list[float]:
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
                if active == 2:
                    two_started.set()
            try:
                assert release.wait(timeout=1)
                return [float(len(query))]
            finally:
                with counter_lock:
                    active -= 1

        service = MagicMock()
        service.embed_query.side_effect = embed_query
        executor = AsyncEmbeddingExecutor(service, max_concurrency=2)

        callers = [asyncio.create_task(executor.embed_query(str(index))) for index in range(6)]
        assert await asyncio.to_thread(two_started.wait, 1)
        release.set()
        results = await asyncio.gather(*callers)

        assert maximum_active == 2
        assert results == [[1.0]] * 6

    def test_rejects_non_positive_concurrency(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            AsyncEmbeddingExecutor(MagicMock(), max_concurrency=0)
