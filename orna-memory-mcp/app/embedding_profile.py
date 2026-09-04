"""Single source of truth for the active dense embedding profile."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Параметры, определяющие совместимость dense embeddings."""

    version: str
    model_name: str
    dimension: int
    query_prefix: str
    passage_prefix: str
    source_repository: str
    source_revision: str
    required_files: tuple[str, ...]

    def snapshot_path(self, cache_dir: Path) -> Path:
        """Возвращает deterministic HF snapshot path для pinned revision."""
        repository_dir = f"models--{self.source_repository.replace('/', '--')}"
        return cache_dir / repository_dir / "snapshots" / self.source_revision


ACTIVE_EMBEDDING_PROFILE = EmbeddingProfile(
    version="e5-v1",
    model_name="intfloat/multilingual-e5-large",
    dimension=1024,
    query_prefix="query: ",
    passage_prefix="passage: ",
    source_repository="qdrant/multilingual-e5-large-onnx",
    source_revision="66076b8dc6e367337e3e90e6fb309fb0f3addaf6",
    required_files=(
        "config.json",
        "model.onnx",
        "model.onnx_data",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
)
