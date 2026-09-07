"""ASGI composition root for the Orna Memory MCP service."""

from app.config import settings
from app.mcp_server import create_http_app

app = create_http_app(settings)
