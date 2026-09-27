"""Eval-only killable E5 process; unchanged EmbeddingService/profile/runtime."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from time import perf_counter


def write_json(stream, payload):
    view = memoryview((json.dumps(payload) + "\n").encode())
    while view:
        count = stream.write(view)
        if not count:
            raise RuntimeError("E5 pipe write failed")
        view = view[count:]


class ProcessEmbeddingExecutor:
    def __init__(self, settings, monitor=None, *, command=None):
        self.monitor = monitor
        self.errors = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            command or [sys.executable, "-m", __name__],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            bufsize=0,
        )
        os.set_blocking(self.process.stdout.fileno(), False)
        self.first = True
        if settings is not None:
            config = {
                k: v
                for k, v in settings.model_dump(mode="json").items()
                if k.startswith("embedding_")
            }
            write_json(self.process.stdin, config)
        if monitor:
            monitor.state["embedding_worker"].value = self.process.pid

    async def _call(self, operation, text):
        started = perf_counter()
        write_json(self.process.stdin, {"operation": operation, "text": text})
        buffer = b""
        while b"\n" not in buffer:
            if self.monitor:
                self.monitor.check()
            if self.process.poll() is not None:
                raise RuntimeError("E5 process exited")
            if perf_counter() - started > (120 if self.first else 180):
                raise RuntimeError("E5 query timeout")
            try:
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if chunk:
                    buffer += chunk
            except BlockingIOError:
                pass
            if b"\n" not in buffer:
                await asyncio.sleep(0.025)
        self.first = False
        reply = json.loads(buffer)
        if "error" in reply:
            raise RuntimeError("E5 process error")
        return reply["vector"]

    async def embed_query(self, text):
        return await self._call("query", text)

    async def embed_memory(self, text):
        return await self._call("memory", text)

    async def aclose(self):
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("E5 worker cleanup timeout") from exc
        finally:
            self.process.stdin.close()
            self.process.stdout.close()
            self.errors.close()
            if self.monitor:
                self.monitor.state["embedding_worker"].value = 0


def main():
    from unittest.mock import patch

    with patch("pydantic_settings.sources.DotEnvSettingsSource._read_env_files", return_value={}):
        from app.config import Settings
        from app.embeddings import EmbeddingService
    config = json.loads(sys.stdin.readline())
    service = EmbeddingService(Settings(_env_file=None, **config))
    for line in sys.stdin:
        request = json.loads(line)
        method = service.embed_query if request["operation"] == "query" else service.embed_memory
        try:
            reply = {"vector": method(request["text"])}
        except Exception as exc:
            reply = {"error": type(exc).__name__}
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
