from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

import app.mcp_server as mcp_server_module
from app.config import Settings
from app.mcp_server import create_http_app, create_mcp_server


def _settings(token: str) -> Settings:
    return Settings(orna_memory_token=token, _env_file=None)


@pytest.mark.parametrize("configured_token", ["", " ", "\t\n"])
def test_server_composition_fails_fast_for_blank_token(configured_token):
    with pytest.raises(ValueError, match="configured bearer token must not be blank"):
        create_mcp_server(_settings(configured_token))


def test_http_app_owns_production_database_pool_lifecycle(monkeypatch):
    config = _settings("correct-token")
    pool = AsyncMock()
    create_db_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr(mcp_server_module, "create_db_pool", create_db_pool)
    length_guard = MagicMock()
    monkeypatch.setattr(mcp_server_module, "E5LengthGuard", length_guard)
    search_service = MagicMock()
    search_service_factory = MagicMock(return_value=search_service)
    monkeypatch.setattr(mcp_server_module, "MemorySearchService", search_service_factory)
    app = create_http_app(config)

    with TestClient(app, base_url="http://127.0.0.1:8000"):
        create_db_pool.assert_awaited_once_with(config)

    pool.close.assert_awaited_once_with()
    length_guard.assert_called_once_with(config)
    search_service_factory.assert_called_once()
    repository, embeddings, search_config = search_service_factory.call_args.args
    assert repository._pool is pool
    assert embeddings._service.model_name == config.embedding_model
    assert search_config is config


def test_http_app_drains_embedding_executor_before_closing_database_pool(monkeypatch):
    config = _settings("correct-token")
    events: list[str] = []
    pool = AsyncMock()

    async def close_pool():
        events.append("pool")

    pool.close.side_effect = close_pool
    monkeypatch.setattr(mcp_server_module, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(mcp_server_module, "E5LengthGuard", MagicMock())

    executor = MagicMock()

    async def close_executor():
        events.append("executor")

    executor.aclose = AsyncMock(side_effect=close_executor)
    monkeypatch.setattr(
        mcp_server_module,
        "AsyncEmbeddingExecutor",
        MagicMock(return_value=executor),
    )

    app = create_http_app(config)
    with TestClient(app, base_url="http://127.0.0.1:8000"):
        pass

    executor.aclose.assert_awaited_once_with()
    assert events == ["executor", "pool"]
