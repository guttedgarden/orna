"""Authenticated Streamable HTTP MCP server composition."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.server.mcpserver.tools import Tool
from mcp_types import ToolAnnotations
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette

from app.auth import StaticBearerTokenVerifier
from app.config import Settings
from app.db import create_db_pool
from app.embeddings import AsyncEmbeddingExecutor, EmbeddingService
from app.models import MemoryScope, MemoryStatus
from app.project_context import ProjectHeaderError, resolve_project_header
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService
from app.write import MemoryAddCommand, MemoryWriteService
from app.write_safety import E5LengthGuard, MemorySafetyError, MemoryWriteSafety

MCP_PATH = "/mcp"
INVALID_TOOL_ARGUMENTS_MESSAGE = "invalid tool arguments"


class StrictSafeTool(Tool):
    """Строгий MCP tool, не отражающий невалидные аргументы в публичной ошибке."""

    async def run(
        self,
        arguments: dict[str, Any],
        context: Context[Any, Any],
        convert_result: bool = False,
    ) -> Any:
        try:
            return await super().run(arguments, context, convert_result)
        except ToolError as exc:
            # В mcp 2.1.1 только argument ValidationError оборачивается напрямую в
            # ToolError. Ошибки handler оборачиваются через собственный ToolError,
            # а ошибки public output — через UnexpectedToolError.
            if isinstance(exc.__cause__, ValidationError) and not isinstance(
                exc, UnexpectedToolError
            ):
                raise ToolError(INVALID_TOOL_ARGUMENTS_MESSAGE) from None
            raise


def _create_strict_safe_tool(
    handler: Callable[..., Any],
    *,
    name: str,
    annotations: ToolAnnotations,
) -> Tool:
    """Создаёт переиспользуемый MCP tool со строгой схемой и безопасной ошибкой."""
    tool = StrictSafeTool.from_function(
        handler,
        name=name,
        annotations=annotations,
        structured_output=True,
    )
    argument_model = tool.fn_metadata.arg_model
    argument_model.model_config.update(extra="forbid", hide_input_in_errors=True)
    argument_model.model_rebuild(force=True)
    tool.parameters = argument_model.model_json_schema(by_alias=True)
    return tool


@dataclass(frozen=True, slots=True)
class MCPDependencies:
    """Request-independent dependencies с lifecycle, принадлежащим MCP server."""

    repository: MemoryRepository
    write_service: MemoryWriteService | None
    search_service: MemorySearchService | None


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


class MemorySearchResultItem(BaseModel):
    """Public search item без ranking diagnostics и storage internals."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    id: UUID
    logical_id: UUID
    revision: int = Field(ge=1)
    scope: MemoryScope
    project_id: str | None
    memory_type: str
    status: MemoryStatus
    content: str
    tags: list[str]
    identifiers: list[str]
    provenance: dict[str, Any]


class MemorySearchResponse(BaseModel):
    """Stable public wrapper for MCP search results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: list[MemorySearchResultItem]


def _create_lifespan(
    config: Settings,
    repository: MemoryRepository | None,
    write_service: MemoryWriteService | None,
    search_service: MemorySearchService | None,
) -> Callable[
    [MCPServer[MCPDependencies]],
    AbstractAsyncContextManager[MCPDependencies],
]:
    @asynccontextmanager
    async def lifespan(_server: MCPServer[MCPDependencies]) -> AsyncIterator[MCPDependencies]:
        if repository is not None:
            yield MCPDependencies(
                repository=repository,
                write_service=write_service,
                search_service=search_service,
            )
            return

        pool = await create_db_pool(config)
        embeddings: AsyncEmbeddingExecutor | None = None
        try:
            production_repository = MemoryRepository(pool, config)
            embedding_service = EmbeddingService(config)
            embeddings = AsyncEmbeddingExecutor(
                embedding_service,
                max_concurrency=config.embedding_max_concurrency,
            )
            safety = MemoryWriteSafety(E5LengthGuard(config))
            yield MCPDependencies(
                repository=production_repository,
                write_service=MemoryWriteService(
                    production_repository,
                    embeddings,
                    config,
                    safety,
                ),
                search_service=MemorySearchService(
                    production_repository,
                    embeddings,
                    config,
                ),
            )
        finally:
            shutdown_error: BaseException | None = None
            try:
                if embeddings is not None:
                    await embeddings.aclose()
            except BaseException as error:
                shutdown_error = error

            pool_close = asyncio.create_task(pool.close())
            while not pool_close.done():
                try:
                    await asyncio.shield(pool_close)
                except asyncio.CancelledError as error:
                    if shutdown_error is None:
                        shutdown_error = error
                except BaseException:
                    break

            pool_error: BaseException | None = None
            try:
                pool_close.result()
            except BaseException as error:
                pool_error = error

            if shutdown_error is not None:
                if pool_error is not None:
                    raise shutdown_error from pool_error
                raise shutdown_error
            if pool_error is not None:
                raise pool_error

    return lifespan


def _create_memory_get_tool() -> Tool:
    async def memory_get(
        memory_id: Annotated[
            UUID,
            Field(
                description=(
                    "Physical UUID returned by memory_search or memory_add. Use this tool when "
                    "the exact revision, status, timestamps, or provenance must be inspected."
                )
            ),
        ],
        ctx: Context[MCPDependencies, Any],
    ) -> MemoryGetResult:
        """Read one exact memory revision by physical UUID.

        Use after memory_search or memory_add when the exact revision, lifecycle status,
        timestamps, or provenance matters. This is not a search tool: pass a returned physical
        memory id. Visibility remains limited to the current project plus global memories.
        """
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

    return _create_strict_safe_tool(
        memory_get,
        name="memory_get",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )


def _create_memory_add_tool() -> Tool:
    async def memory_add(
        content: Annotated[
            str,
            Field(
                description=(
                    "A concise, self-contained durable claim. Include the decision or finding "
                    "and the reason it will matter in future work."
                )
            ),
        ],
        memory_type: Annotated[
            str,
            Field(
                description=(
                    "Client-defined category such as decision, incident, constraint, "
                    "architecture, or convention."
                )
            ),
        ],
        # MCP 2.1.1 берёт default для public schema прямо из сигнатуры функции.
        # Эти списки не изменяются: MemoryAddCommand создаёт собственные копии.
        tags: Annotated[
            list[str],
            Field(description="Optional short topic labels that improve later retrieval."),
        ] = [],  # noqa: B006
        identifiers: Annotated[
            list[str],
            Field(
                description=(
                    "Optional exact technical names such as classes, functions, commands, "
                    "error codes, configuration keys, or file paths."
                )
            ),
        ] = [],  # noqa: B006
        *,
        ctx: Context[MCPDependencies, Any],
    ) -> MemoryGetResult:
        """Store confirmed, reusable experience for future work in the current project.

        Use after completing or diagnosing work when the result is durable and non-obvious:
        a decision with rationale, an incident root cause or workaround, an invariant, a
        constraint, or a project convention. Search first when practical to avoid duplicates.
        Do not store raw logs, transient task state, speculation, easily rediscoverable code,
        or credentials. Records are always project-scoped. Summarize long text into one concise,
        self-contained claim before writing.
        """
        try:
            project_id = resolve_project_header(ctx.headers)
        except ProjectHeaderError as exc:
            raise ToolError(str(exc)) from exc

        write_service = ctx.request_context.lifespan_context.write_service
        if write_service is None:
            raise ToolError("memory writes are unavailable")

        try:
            command = MemoryAddCommand(
                content=content,
                scope=MemoryScope.PROJECT,
                memory_type=memory_type,
                tags=tags,
                identifiers=identifiers,
                provenance={
                    "created_by": "codex",
                    "source": {"kind": "agent_explicit_add"},
                    "project_id": project_id,
                },
            )
            record = await write_service.add(command, project_id=project_id)
        except ValidationError as exc:
            raise ToolError("invalid memory arguments") from exc
        except MemorySafetyError as exc:
            raise ToolError(str(exc)) from exc

        return MemoryGetResult.model_validate(record)

    return _create_strict_safe_tool(
        memory_add,
        name="memory_add",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )


def _create_memory_search_tool() -> Tool:
    async def memory_search(
        query: Annotated[
            str,
            Field(
                description=(
                    "Natural-language search describing the current task, symptom, error, "
                    "component, or decision. Include exact identifiers when available."
                )
            ),
        ],
        memory_type: Annotated[
            str | None,
            Field(
                description=(
                    "Optional exact client-defined category filter. Omit it when relevant "
                    "experience may have different categories."
                )
            ),
        ] = None,
        *,
        ctx: Context[MCPDependencies, Any],
    ) -> MemorySearchResponse:
        """Search prior durable experience before substantial work or troubleshooting.

        Use early when previous decisions, incidents, constraints, conventions, or workarounds
        may affect the task. Search with the task, symptom, error text, component names, and exact
        identifiers rather than asking whether any memory exists. Results include applicable
        global memories, can be limited by memory_type, and contain at most 5 items. Treat results
        as context to verify against current code and authoritative documentation, not as a
        replacement for them.
        """
        try:
            project_id = resolve_project_header(ctx.headers)
        except ProjectHeaderError as exc:
            raise ToolError(str(exc)) from exc

        search_service = ctx.request_context.lifespan_context.search_service
        if search_service is None:
            raise ToolError("memory search is unavailable")

        try:
            search_query = MemorySearchQuery(
                query=query,
                memory_type=memory_type,
                limit=5,
            )
        except ValidationError as exc:
            raise ToolError("invalid memory search arguments") from exc

        results = await search_service.search(search_query, project_id=project_id)
        return MemorySearchResponse(
            results=[MemorySearchResultItem.model_validate(result) for result in results]
        )

    return _create_strict_safe_tool(
        memory_search,
        name="memory_search",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )


def create_mcp_server(
    config: Settings,
    *,
    repository: MemoryRepository | None = None,
    write_service: MemoryWriteService | None = None,
    search_service: MemorySearchService | None = None,
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
        tools=[
            _create_memory_get_tool(),
            _create_memory_add_tool(),
            _create_memory_search_tool(),
        ],
        lifespan=_create_lifespan(config, repository, write_service, search_service),
    )


def create_http_app(
    config: Settings,
    *,
    repository: MemoryRepository | None = None,
    write_service: MemoryWriteService | None = None,
    search_service: MemorySearchService | None = None,
) -> Starlette:
    """Собирает JSON-response Streamable HTTP app без transport session state."""
    server = create_mcp_server(
        config,
        repository=repository,
        write_service=write_service,
        search_service=search_service,
    )
    return server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,
        stateless_http=True,
        host=config.mcp_host,
    )
