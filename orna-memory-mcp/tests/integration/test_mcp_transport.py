from starlette.testclient import TestClient

from app.config import Settings
from app.mcp_server import create_http_app

_PROTOCOL_VERSION = "2026-07-28"


def _tools_list_request() -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": _PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": {
                    "name": "orna-memory-test",
                    "version": "1.0",
                },
            }
        },
    }


def _mcp_headers(token: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "MCP-Protocol-Version": _PROTOCOL_VERSION,
        "Mcp-Method": "tools/list",
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _settings(token: str) -> Settings:
    return Settings(orna_memory_token=token, _env_file=None)


def test_missing_and_invalid_tokens_have_same_unauthorized_response():
    configured_token = "configured-secret"
    app = create_http_app(_settings(configured_token))

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        missing = client.post(
            "/mcp",
            json=_tools_list_request(),
            headers=_mcp_headers(),
        )
        invalid = client.post(
            "/mcp",
            json=_tools_list_request(),
            headers=_mcp_headers("presented-secret"),
        )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert (
        missing.json()
        == invalid.json()
        == {
            "error": "invalid_token",
            "error_description": "Authentication required",
        }
    )
    assert missing.headers["www-authenticate"] == invalid.headers["www-authenticate"]
    assert "resource_metadata" not in missing.headers["www-authenticate"]
    assert configured_token not in missing.text
    assert configured_token not in invalid.text
    assert "presented-secret" not in invalid.text


def test_valid_token_reaches_stateless_modern_mcp_handler():
    app = create_http_app(_settings("correct-token"))

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_tools_list_request(),
            headers=_mcp_headers("correct-token"),
        )

    assert response.status_code == 200
    assert response.json()["id"] == 1
    assert response.json()["result"]["tools"] == []
    assert "mcp-session-id" not in response.headers


def test_oauth_metadata_is_not_published_for_static_auth():
    app = create_http_app(_settings("correct-token"))

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.get("/.well-known/oauth-protected-resource/mcp")

    assert response.status_code == 404
