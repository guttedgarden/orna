"""Minimal TCP proxy exposing the internal MCP runtime on host loopback."""

import asyncio
import os
from dataclasses import dataclass
from functools import partial


@dataclass(frozen=True)
class ProxySettings:
    listen_host: str = "0.0.0.0"
    listen_port: int = 8000
    target_host: str = "orna-memory-mcp"
    target_port: int = 8000


def _env_port(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def load_settings() -> ProxySettings:
    """Load the non-secret proxy routing contract from environment variables."""
    return ProxySettings(
        listen_host=os.environ.get("TCP_PROXY_LISTEN_HOST", "0.0.0.0"),
        listen_port=_env_port("TCP_PROXY_LISTEN_PORT", 8000),
        target_host=os.environ.get("TCP_PROXY_TARGET_HOST", "orna-memory-mcp"),
        target_port=_env_port("TCP_PROXY_TARGET_PORT", 8000),
    )


async def _copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(64 * 1024):
        writer.write(data)
        await writer.drain()


async def proxy_connection(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    target_host: str,
    target_port: int,
) -> None:
    """Forward one TCP connection without inspecting HTTP headers or payloads."""
    target_writer: asyncio.StreamWriter | None = None
    tasks: set[asyncio.Task[None]] = set()
    try:
        target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
        tasks = {
            asyncio.create_task(_copy_stream(client_reader, target_writer)),
            asyncio.create_task(_copy_stream(target_reader, client_writer)),
        }
        _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        client_writer.close()
        await client_writer.wait_closed()
        if target_writer is not None:
            target_writer.close()
            await target_writer.wait_closed()


async def start_proxy(settings: ProxySettings) -> asyncio.Server:
    """Create the proxy server without entering its serve loop."""
    handler = partial(
        proxy_connection,
        target_host=settings.target_host,
        target_port=settings.target_port,
    )
    return await asyncio.start_server(handler, settings.listen_host, settings.listen_port)


async def serve(settings: ProxySettings) -> None:
    """Serve TCP connections until the process is stopped."""
    server = await start_proxy(settings)
    print(
        f"MCP loopback proxy listening on {settings.listen_host}:{settings.listen_port} "
        f"and forwarding to {settings.target_host}:{settings.target_port}."
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    """CLI entry point for the Compose loopback proxy service."""
    asyncio.run(serve(load_settings()))


if __name__ == "__main__":
    main()
