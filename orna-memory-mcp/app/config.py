from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Детерминированное определение расположения файлов .env относительно структуры репозитория
_APP_DIR = Path(__file__).resolve().parent
_SERVICE_DIR = _APP_DIR.parent
_REPO_DIR = _SERVICE_DIR.parent

_ENV_FILES: list[Path] = []
if (_REPO_DIR / ".env").is_file():
    _ENV_FILES.append(_REPO_DIR / ".env")
if (_SERVICE_DIR / ".env").is_file():
    _ENV_FILES.append(_SERVICE_DIR / ".env")


class Settings(BaseSettings):
    """Application settings for orna-memory-mcp service."""

    model_config = SettingsConfigDict(
        env_file=tuple(_ENV_FILES) if _ENV_FILES else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding settings
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_threads: int = 2

    # Database settings (PostgreSQL 16 + pgvector)
    # Параметры подключения к PostgreSQL и настройки connection pool
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_user: str = "orna"
    postgres_password: str = ""
    postgres_db: str = "orna_memory"
    database_url: str | None = None
    database_pool_min_size: int = 2
    database_pool_max_size: int = 10

    # Search & retrieval settings
    # Параметры плотного (dense) и гибридного (RRF) поиска
    dense_retrieval_strategy: Literal["exact", "hnsw"] = "exact"
    retrieval_candidate_pool_size: int = 30
    rrf_k: int = 60
    hnsw_ef_search: int = 40
    hnsw_iterative_scan: Literal["off", "strict_order", "relaxed_order"] = "relaxed_order"

    # Profile versions
    # Версии профилей векторизации и лексического анализа
    embedding_profile_version: str = "e5-v1"
    lexical_profile_version: str = "lexical-v1"

    # MCP server settings
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000
    orna_memory_token: str = ""

    @model_validator(mode="after")
    def assemble_database_url_and_validate_pool(self) -> Self:
        # Вычисляем database_url, если он не был передан явно
        if not self.database_url:
            self.database_url = (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )

        # Валидация размера connection pool
        if self.database_pool_max_size < 2:
            raise ValueError("database_pool_max_size must be >= 2")
        if self.database_pool_max_size < self.database_pool_min_size:
            raise ValueError("database_pool_max_size must be >= database_pool_min_size")

        return self


settings = Settings()
