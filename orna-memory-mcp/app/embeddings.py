"""Dense embedding service backed by FastEmbed."""

import math
from collections.abc import Sequence
from threading import Lock
from typing import Any

from fastembed import TextEmbedding

from app.config import Settings, settings
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE, EmbeddingProfile


class EmbeddingOutputError(ValueError):
    """FastEmbed вернул результат, несовместимый с активным profile."""


class ModelCacheMissingError(RuntimeError):
    """Pinned model snapshot отсутствует в configured cache."""


def ensure_prefix(text: str, prefix: str) -> str:
    """Применяет canonical E5 prefix ровно один раз."""
    tag = prefix.split(":", maxsplit=1)[0].strip().lower()
    current = text.strip()

    while current.lower().startswith(f"{tag}:"):
        current = current[len(tag) + 1 :].strip()

    return f"{prefix}{current}"


class EmbeddingService:
    """Вычисляет embeddings только в рамках единственного активного profile."""

    def __init__(self, app_settings: Settings | None = None) -> None:
        configured_settings = app_settings or settings
        self.profile: EmbeddingProfile = ACTIVE_EMBEDDING_PROFILE
        self.model_name = configured_settings.embedding_model
        self.profile_version = configured_settings.embedding_profile_version
        self.threads = configured_settings.embedding_threads
        self.cache_dir = configured_settings.embedding_cache_dir
        self.local_files_only = configured_settings.embedding_local_files_only
        self._model: TextEmbedding | None = None
        self._model_lock = Lock()

        if self.model_name != self.profile.model_name:
            raise ValueError("configured model does not match the active embedding profile")
        if self.profile_version != self.profile.version:
            raise ValueError("configured version does not match the active embedding profile")

        catalog_dimension = TextEmbedding.get_embedding_size(self.model_name)
        if catalog_dimension != self.profile.dimension:
            raise ValueError(
                "FastEmbed model dimension does not match the active embedding profile: "
                f"expected {self.profile.dimension}, got {catalog_dimension}"
            )

    @property
    def model(self) -> TextEmbedding:
        """Потокобезопасно инициализирует одну ONNX model instance на service."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    model_options: dict[str, Any] = {
                        "model_name": self.model_name,
                        "cache_dir": str(self.cache_dir),
                        "threads": self.threads,
                        "local_files_only": self.local_files_only,
                    }
                    if self.local_files_only:
                        snapshot_path = self.profile.snapshot_path(self.cache_dir)
                        if not snapshot_path.is_dir():
                            raise ModelCacheMissingError(
                                "pinned embedding snapshot is missing; run "
                                "`python -m app.model_cache` before starting the service"
                            )
                        model_options["specific_model_path"] = str(snapshot_path)

                    self._model = TextEmbedding(
                        **model_options,
                    )
        return self._model

    def embed_query(self, query: str) -> list[float]:
        """Векторизует один query с canonical ``query:`` prefix."""
        prepared_query = ensure_prefix(query, self.profile.query_prefix)
        return self._embed_prepared([prepared_query])[0]

    def embed_memory(self, content: str) -> list[float]:
        """Векторизует одну memory с canonical ``passage:`` prefix."""
        prepared_content = ensure_prefix(content, self.profile.passage_prefix)
        return self._embed_prepared([prepared_content])[0]

    def embed_memories(self, contents: Sequence[str]) -> list[list[float]]:
        """Векторизует batch memories, не принимая одиночную строку как sequence."""
        if isinstance(contents, (str, bytes)):
            raise TypeError("contents must be a sequence of strings, not a single string")
        if not all(isinstance(content, str) for content in contents):
            raise TypeError("every content item must be a string")
        if not contents:
            return []

        prepared = [ensure_prefix(content, self.profile.passage_prefix) for content in contents]
        return self._embed_prepared(prepared)

    def _embed_prepared(self, prepared_texts: list[str]) -> list[list[float]]:
        vectors = list(self.model.embed(prepared_texts))
        if len(vectors) != len(prepared_texts):
            raise EmbeddingOutputError(
                "FastEmbed returned an unexpected number of embeddings: "
                f"expected {len(prepared_texts)}, got {len(vectors)}"
            )
        return [self._validate_vector(vector) for vector in vectors]

    def _validate_vector(self, vector: Any) -> list[float]:
        raw_values = vector.tolist() if hasattr(vector, "tolist") else vector
        if not isinstance(raw_values, list):
            raise EmbeddingOutputError("FastEmbed embedding must be a one-dimensional vector")

        try:
            values = [float(value) for value in raw_values]
        except (TypeError, ValueError) as error:
            raise EmbeddingOutputError("FastEmbed embedding must contain numbers") from error

        if len(values) != self.profile.dimension:
            raise EmbeddingOutputError(
                "FastEmbed embedding dimension does not match the active profile: "
                f"expected {self.profile.dimension}, got {len(values)}"
            )
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingOutputError("FastEmbed embedding values must be finite")
        if not any(value != 0.0 for value in values):
            raise EmbeddingOutputError("FastEmbed embedding must have a non-zero norm")

        return values


embedding_service = EmbeddingService()
