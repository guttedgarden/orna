import pytest
from pydantic import ValidationError

from app.config import Settings


class TestConfigDefaults:
    """Проверка значений конфигурации по умолчанию / Verify default settings values."""

    def test_default_database_settings(self, monkeypatch: pytest.MonkeyPatch):
        # Очищаем env variables, чтобы проверить чистые defaults
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

        cfg = Settings(_env_file=None)
        assert cfg.postgres_host == "127.0.0.1"
        assert cfg.postgres_port == 5432
        assert cfg.postgres_user == "orna"
        assert cfg.postgres_password == ""
        assert cfg.postgres_db == "orna_memory"
        assert cfg.database_pool_min_size == 2
        assert cfg.database_pool_max_size == 10

    def test_default_search_settings(self):
        cfg = Settings(_env_file=None)
        assert cfg.dense_retrieval_strategy == "exact"
        assert cfg.retrieval_candidate_pool_size == 20
        assert cfg.rrf_k == 60
        assert cfg.hnsw_ef_search == 40
        assert cfg.hnsw_iterative_scan == "relaxed_order"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"retrieval_candidate_pool_size": 0},
            {"rrf_k": 0},
            {"hnsw_ef_search": 0},
        ],
    )
    def test_rejects_non_positive_search_settings(self, overrides):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Settings(**overrides, _env_file=None)

    def test_default_profile_versions(self):
        cfg = Settings(_env_file=None)
        assert cfg.embedding_profile_version == "e5-v1"
        assert cfg.embedding_max_concurrency == 1
        assert cfg.embedding_cache_dir.name == "fastembed"
        assert cfg.embedding_local_files_only is True
        assert cfg.lexical_profile_version == "lexical-v1"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"embedding_model": "BAAI/bge-small-en-v1.5"}, "embedding_model"),
            ({"embedding_profile_version": "e5-v2"}, "embedding_profile_version"),
            ({"embedding_threads": 0}, "greater than or equal to 1"),
            ({"embedding_max_concurrency": 0}, "greater than or equal to 1"),
        ],
    )
    def test_rejects_incompatible_embedding_settings(self, overrides, message):
        with pytest.raises(ValidationError, match=message):
            Settings(**overrides, _env_file=None)


class TestDatabaseUrlComputation:
    """Проверка формирования database_url / Verify database_url computation and overrides."""

    def test_computed_database_url_from_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

        cfg = Settings(_env_file=None)
        assert cfg.database_url == "postgresql://orna:@127.0.0.1:5432/orna_memory"

    def test_computed_database_url_with_password(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("POSTGRES_PASSWORD", "secret_pass")
        monkeypatch.delenv("DATABASE_URL", raising=False)

        cfg = Settings(_env_file=None)
        assert cfg.database_url == "postgresql://orna:secret_pass@127.0.0.1:5432/orna_memory"

    def test_computed_database_url_with_custom_params(self):
        cfg = Settings(
            postgres_host="db.internal",
            postgres_port=5433,
            postgres_user="custom_user",
            postgres_password="custom_password",
            postgres_db="custom_db",
            _env_file=None,
        )
        assert (
            cfg.database_url
            == "postgresql://custom_user:custom_password@db.internal:5433/custom_db"
        )

    def test_explicit_database_url_preserved(self):
        custom_url = "postgresql://override_user:override_pass@remote_host:5439/override_db"
        cfg = Settings(database_url=custom_url, _env_file=None)
        assert cfg.database_url == custom_url


class TestConnectionPoolValidation:
    """Проверка валидации connection pool / Verify connection pool constraints."""

    def test_valid_pool_sizes(self):
        # Минимально допустимый max_size (2) и равный min_size
        cfg = Settings(database_pool_min_size=2, database_pool_max_size=2, _env_file=None)
        assert cfg.database_pool_min_size == 2
        assert cfg.database_pool_max_size == 2

    def test_max_size_less_than_two_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(database_pool_max_size=1, _env_file=None)
        assert "database_pool_max_size must be >= 2" in str(exc_info.value)

    def test_max_size_less_than_min_size_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(database_pool_min_size=5, database_pool_max_size=3, _env_file=None)
        assert "database_pool_max_size must be >= database_pool_min_size" in str(exc_info.value)
