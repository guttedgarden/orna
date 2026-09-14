import asyncio

from app.tcp_proxy import ProxySettings, start_proxy


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        writer.write(await reader.read(64 * 1024))
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def test_proxy_forwards_bytes_without_inspecting_payload():
    target = await asyncio.start_server(_echo, "127.0.0.1", 0)
    target_port = target.sockets[0].getsockname()[1]
    proxy = await start_proxy(
        ProxySettings(
            listen_host="127.0.0.1",
            listen_port=0,
            target_host="127.0.0.1",
            target_port=target_port,
        )
    )
    proxy_port = proxy.sockets[0].getsockname()[1]

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(b"Authorization: Bearer opaque\r\n\r\npayload")
        await writer.drain()

        assert await reader.read() == b"Authorization: Bearer opaque\r\n\r\npayload"
        writer.close()
        await writer.wait_closed()
    finally:
        proxy.close()
        target.close()
        await proxy.wait_closed()
        await target.wait_closed()
