"""Single source of truth for the active dense embedding profile."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Параметры, определяющие совместимость dense embeddings."""

    version: str
    model_name: str
    dimension: int
    query_prefix: str
    passage_prefix: str


ACTIVE_EMBEDDING_PROFILE = EmbeddingProfile(
    version="e5-v1",
    model_name="intfloat/multilingual-e5-large",
    dimension=1024,
    query_prefix="query: ",
    passage_prefix="passage: ",
)
