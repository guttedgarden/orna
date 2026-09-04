import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import (
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
