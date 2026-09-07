"""Authenticated Streamable HTTP MCP server composition."""

from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from pydantic import AnyHttpUrl
from starlette.applications import Starlette

from app.auth import StaticBearerTokenVerifier
from app.config import Settings

MCP_PATH = "/mcp"


def create_mcp_server(config: Settings) -> MCPServer[None]:
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
    )


def create_http_app(config: Settings) -> Starlette:
    """Собирает JSON-response Streamable HTTP app без transport session state."""
    server = create_mcp_server(config)
    return server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,
        stateless_http=True,
        host=config.mcp_host,
    )
