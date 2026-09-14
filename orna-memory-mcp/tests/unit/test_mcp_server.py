import asyncio
from threading import Event
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


async def test_cancelled_shutdown_drains_inference_before_closing_pool(monkeypatch):
    config = _settings("correct-token")
    inference_started = Event()
    release_inference = Event()
    inference_finished = Event()
    pool_close_started = asyncio.Event()
    release_pool_close = asyncio.Event()
    pool_close_finished = asyncio.Event()
    pool_close_cancelled = False
    events: list[str] = []

    def embed_query(_query: str) -> list[float]:
        inference_started.set()
        try:
            assert release_inference.wait(timeout=1)
            return [1.0]
        finally:
            inference_finished.set()

    embedding_service = MagicMock()
    embedding_service.embed_query.side_effect = embed_query
    monkeypatch.setattr(
        mcp_server_module,
        "EmbeddingService",
        MagicMock(return_value=embedding_service),
    )
    monkeypatch.setattr(mcp_server_module, "E5LengthGuard", MagicMock())

    pool = AsyncMock()

    async def close_pool():
        nonlocal pool_close_cancelled
        assert inference_finished.is_set()
        events.append("pool-started")
        pool_close_started.set()
        try:
            await release_pool_close.wait()
        except asyncio.CancelledError:
            pool_close_cancelled = True
            raise
        events.append("pool-finished")
        pool_close_finished.set()

    pool.close.side_effect = close_pool
    monkeypatch.setattr(mcp_server_module, "create_db_pool", AsyncMock(return_value=pool))

    lifespan = mcp_server_module._create_lifespan(config, None, None, None)
    context = lifespan(MagicMock())
    dependencies = await context.__aenter__()
    executor = dependencies.search_service._embeddings
    inference = asyncio.create_task(executor.embed_query("blocked"))
    shutdown: asyncio.Task[object] | None = None

    try:
        assert await asyncio.to_thread(inference_started.wait, 1)
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        await asyncio.sleep(0)
        shutdown.cancel("lifespan shutdown cancellation")
        await asyncio.sleep(0)

        assert not shutdown.done()
        assert not inference_finished.is_set()
        assert len(executor._inflight) == 1
        pool.close.assert_not_awaited()

        release_inference.set()
        assert await inference == [1.0]
        await asyncio.wait_for(pool_close_started.wait(), timeout=1)
        assert not shutdown.done()
        assert not pool_close_finished.is_set()

        shutdown.cancel("second lifespan shutdown cancellation")
        await asyncio.sleep(0)
        assert not shutdown.done()
        assert not pool_close_cancelled

        release_pool_close.set()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await shutdown

        assert cancellation.value.args == ("lifespan shutdown cancellation",)
        assert executor._inflight == set()
        pool.close.assert_awaited_once_with()
        assert events == ["pool-started", "pool-finished"]
    finally:
        release_inference.set()
        release_pool_close.set()
        cleanup = [inference]
        if shutdown is not None:
            cleanup.append(shutdown)
        await asyncio.gather(*cleanup, return_exceptions=True)


async def test_pool_closes_when_embedding_executor_shutdown_fails(monkeypatch):
    config = _settings("correct-token")
    pool = AsyncMock()
    monkeypatch.setattr(mcp_server_module, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(mcp_server_module, "E5LengthGuard", MagicMock())

    executor = MagicMock()
    executor.aclose = AsyncMock(side_effect=RuntimeError("executor shutdown failed"))
    monkeypatch.setattr(
        mcp_server_module,
        "AsyncEmbeddingExecutor",
        MagicMock(return_value=executor),
    )

    lifespan = mcp_server_module._create_lifespan(config, None, None, None)
    context = lifespan(MagicMock())
    await context.__aenter__()

    with pytest.raises(RuntimeError, match="executor shutdown failed"):
        await context.__aexit__(None, None, None)

    executor.aclose.assert_awaited_once_with()
    pool.close.assert_awaited_once_with()


async def test_shutdown_cancellation_is_preserved_when_pool_close_fails(monkeypatch):
    config = _settings("correct-token")
    pool = AsyncMock()
    pool.close.side_effect = RuntimeError("pool close failed")
    monkeypatch.setattr(mcp_server_module, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(mcp_server_module, "E5LengthGuard", MagicMock())

    executor = MagicMock()
    executor.aclose = AsyncMock(
        side_effect=asyncio.CancelledError("original shutdown cancellation")
    )
    monkeypatch.setattr(
        mcp_server_module,
        "AsyncEmbeddingExecutor",
        MagicMock(return_value=executor),
    )

    lifespan = mcp_server_module._create_lifespan(config, None, None, None)
    context = lifespan(MagicMock())
    await context.__aenter__()

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await context.__aexit__(None, None, None)

    assert cancellation.value.args == ("original shutdown cancellation",)
    assert isinstance(cancellation.value.__cause__, RuntimeError)
    assert str(cancellation.value.__cause__) == "pool close failed"
    executor.aclose.assert_awaited_once_with()
    pool.close.assert_awaited_once_with()
