"""Authenticated Streamable HTTP MCP server composition."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool
from mcp_types import ToolAnnotations
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field
from starlette.applications import Starlette

from app.auth import StaticBearerTokenVerifier
from app.config import Settings
from app.db import create_db_pool
from app.models import MemoryScope, MemoryStatus
from app.project_context import ProjectHeaderError, resolve_project_header
from app.repository import MemoryRepository

MCP_PATH = "/mcp"


@dataclass(frozen=True, slots=True)
class MCPDependencies:
    """Request-independent dependencies с lifecycle, принадлежащим MCP server."""

    repository: MemoryRepository


class MemoryGetResult(BaseModel):
    """Public MCP representation без embedding и внутренних search-полей."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    id: UUID
    logical_id: UUID
    revision: int = Field(ge=1)
    supersedes_id: UUID | None
    scope: MemoryScope
    project_id: str | None
    memory_type: str
    status: MemoryStatus
    content: str
    tags: list[str]
    identifiers: list[str]
    provenance: dict[str, Any]
    created_at: datetime
    status_changed_at: datetime | None


def _create_lifespan(
    config: Settings,
    repository: MemoryRepository | None,
) -> Callable[
    [MCPServer[MCPDependencies]],
    AbstractAsyncContextManager[MCPDependencies],
]:
    @asynccontextmanager
    async def lifespan(_server: MCPServer[MCPDependencies]) -> AsyncIterator[MCPDependencies]:
        if repository is not None:
            yield MCPDependencies(repository=repository)
            return

        pool = await create_db_pool(config)
        try:
            yield MCPDependencies(repository=MemoryRepository(pool, config))
        finally:
            await pool.close()

    return lifespan


def _create_memory_get_tool() -> Tool:
    async def memory_get(
        memory_id: UUID,
        ctx: Context[MCPDependencies, Any],
    ) -> MemoryGetResult:
        """Read one visible memory revision by its physical UUID."""
        try:
            project_id = resolve_project_header(ctx.headers)
        except ProjectHeaderError as exc:
            raise ToolError(str(exc)) from exc

        record = await ctx.request_context.lifespan_context.repository.get_by_id(
            memory_id,
            project_id,
        )
        if record is None:
            raise ToolError("memory not found")
        return MemoryGetResult.model_validate(record)

    tool = Tool.from_function(
        memory_get,
        name="memory_get",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        structured_output=True,
    )

    # mcp 2.1.1 генерирует argument model с extra='ignore'. Tool contract Orna строгий:
    # неизвестные поля должны отклоняться до вызова handler и repository.
    argument_model = tool.fn_metadata.arg_model
    argument_model.model_config["extra"] = "forbid"
    argument_model.model_rebuild(force=True)
    tool.parameters = argument_model.model_json_schema(by_alias=True)
    return tool


def create_mcp_server(
    config: Settings,
    *,
    repository: MemoryRepository | None = None,
) -> MCPServer[MCPDependencies]:
    """Создаёт MCP server с обязательной static Bearer authentication."""
    verifier = StaticBearerTokenVerifier(config.orna_memory_token)

    # В mcp 2.1.1 AuthSettings включает официальный BearerAuthBackend.
    # resource_server_url намеренно не задан: OAuth metadata появится только вместе с OAuth.
    auth = AuthSettings(
        issuer_url=AnyHttpUrl(f"http://{config.mcp_host}:{config.mcp_port}"),
        resource_server_url=None,
        required_scopes=[],
    )
    return MCPServer(
        "orna-memory",
        token_verifier=verifier,
        auth=auth,
        tools=[_create_memory_get_tool()],
        lifespan=_create_lifespan(config, repository),
    )


def create_http_app(
    config: Settings,
    *,
    repository: MemoryRepository | None = None,
) -> Starlette:
    """Собирает JSON-response Streamable HTTP app без transport session state."""
    server = create_mcp_server(config, repository=repository)
    return server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,
        stateless_http=True,
        host=config.mcp_host,
    )
