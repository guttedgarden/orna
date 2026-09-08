from datetime import UTC, datetime
from unittest.mock import AsyncMock, call
from uuid import UUID, uuid4

from starlette.testclient import TestClient

from app.config import Settings
from app.mcp_server import create_http_app
from app.models import EMBEDDING_DIMENSION, MemoryRecord, MemoryScope, MemoryStatus
from app.repository import MemoryRepository

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


def _tools_call_request(memory_id: str, **extra_arguments: object) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "memory_get",
            "arguments": {"memory_id": memory_id, **extra_arguments},
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": _PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": {
                    "name": "orna-memory-test",
                    "version": "1.0",
                },
            },
        },
    }


def _mcp_headers(
    token: str | None = None,
    *,
    method: str = "tools/list",
    project_id: str | None = None,
) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "MCP-Protocol-Version": _PROTOCOL_VERSION,
        "Mcp-Method": method,
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if method == "tools/call":
        headers["Mcp-Name"] = "memory_get"
    if project_id is not None:
        headers["X-Memory-Project"] = project_id
    return headers


def _settings(token: str) -> Settings:
    return Settings(orna_memory_token=token, _env_file=None)


def _repository() -> AsyncMock:
    return AsyncMock(spec=MemoryRepository)


def _memory_record() -> MemoryRecord:
    memory_id = UUID("019cff03-d6db-7772-89b8-e18dc19a9038")
    return MemoryRecord(
        id=memory_id,
        logical_id=UUID("019cff03-d6db-7772-89b8-e18dc19a9039"),
        revision=2,
        supersedes_id=UUID("019cff03-d6db-7772-89b8-e18dc19a9037"),
        scope=MemoryScope.PROJECT,
        project_id="project-a",
        memory_type="decision",
        status=MemoryStatus.ARCHIVED,
        content="Preserve historical memory revisions.",
        content_hash=b"h" * 32,
        tags=["mcp"],
        identifiers=["memory_get"],
        lexical_source="Preserve historical memory revisions memory_get memory get",
        lexical_profile_version="lexical-v1",
        embedding=[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
        embedding_model="intfloat/multilingual-e5-large",
        embedding_profile_version="e5-v1",
        provenance={"source": "integration-test"},
        created_at=datetime(2026, 9, 8, 12, 0, tzinfo=UTC),
        status_changed_at=datetime(2026, 9, 8, 13, 0, tzinfo=UTC),
    )


def test_missing_and_invalid_tokens_have_same_unauthorized_response():
    configured_token = "configured-secret"
    app = create_http_app(_settings(configured_token), repository=_repository())

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
    app = create_http_app(_settings("correct-token"), repository=_repository())

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_tools_list_request(),
            headers=_mcp_headers("correct-token"),
        )

    assert response.status_code == 200, response.text
    assert response.json()["id"] == 1
    tools = response.json()["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["memory_get"]
    assert tools[0]["inputSchema"]["additionalProperties"] is False
    assert tools[0]["inputSchema"]["required"] == ["memory_id"]
    assert tools[0]["inputSchema"]["properties"]["memory_id"]["format"] == "uuid"
    assert tools[0]["annotations"]["readOnlyHint"] is True
    assert "embedding" not in tools[0]["outputSchema"]["properties"]
    assert "lexical_source" not in tools[0]["outputSchema"]["properties"]
    assert "content_hash" not in tools[0]["outputSchema"]["properties"]
    assert "mcp-session-id" not in response.headers


def test_oauth_metadata_is_not_published_for_static_auth():
    app = create_http_app(_settings("correct-token"), repository=_repository())

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.get("/.well-known/oauth-protected-resource/mcp")

    assert response.status_code == 404


def test_memory_get_reads_project_header_and_returns_structured_record_without_embedding():
    repository = _repository()
    record = _memory_record()
    repository.get_by_id.return_value = record
    app = create_http_app(_settings("correct-token"), repository=repository)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_tools_call_request(str(record.id)),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
            ),
        )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"] == {
        "id": str(record.id),
        "logical_id": str(record.logical_id),
        "revision": 2,
        "supersedes_id": str(record.supersedes_id),
        "scope": "project",
        "project_id": "project-a",
        "memory_type": "decision",
        "status": "archived",
        "content": "Preserve historical memory revisions.",
        "tags": ["mcp"],
        "identifiers": ["memory_get"],
        "provenance": {"source": "integration-test"},
        "created_at": "2026-09-08T12:00:00Z",
        "status_changed_at": "2026-09-08T13:00:00Z",
    }
    repository.get_by_id.assert_awaited_once_with(record.id, "project-a")
    assert "mcp-session-id" not in response.headers


def test_memory_get_rejects_missing_project_header_without_repository_call():
    repository = _repository()
    app = create_http_app(_settings("correct-token"), repository=repository)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_tools_call_request(str(uuid4())),
            headers=_mcp_headers("correct-token", method="tools/call"),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert "X-Memory-Project header is required" in result["content"][0]["text"]
    repository.get_by_id.assert_not_awaited()


def test_memory_get_returns_same_not_found_for_missing_or_hidden_record():
    repository = _repository()
    repository.get_by_id.return_value = None
    app = create_http_app(_settings("correct-token"), repository=repository)
    missing_id = uuid4()

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        missing = client.post(
            "/mcp",
            json=_tools_call_request(str(missing_id)),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
            ),
        )
        hidden = client.post(
            "/mcp",
            json=_tools_call_request(str(missing_id)),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-b",
            ),
        )

    missing_result = missing.json()["result"]
    hidden_result = hidden.json()["result"]
    assert missing_result == hidden_result
    assert missing_result["isError"] is True
    assert missing_result["content"][0]["text"].endswith("memory not found")
    assert repository.get_by_id.await_args_list == [
        call(missing_id, "project-a"),
        call(missing_id, "project-b"),
    ]


def test_memory_get_rejects_extra_arguments_before_repository_call():
    repository = _repository()
    app = create_http_app(_settings("correct-token"), repository=repository)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_tools_call_request(str(uuid4()), project_id="project-b"),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert "project_id" in result["content"][0]["text"]
    repository.get_by_id.assert_not_awaited()


def test_memory_get_rejects_invalid_uuid_before_repository_call():
    repository = _repository()
    app = create_http_app(_settings("correct-token"), repository=repository)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_tools_call_request("not-a-uuid"),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert "memory_id" in result["content"][0]["text"]
    repository.get_by_id.assert_not_awaited()


def test_memory_get_rejects_duplicate_project_header_without_repository_call():
    repository = _repository()
    app = create_http_app(_settings("correct-token"), repository=repository)
    headers = list(
        _mcp_headers(
            "correct-token",
            method="tools/call",
        ).items()
    )
    headers.extend(
        [
            ("X-Memory-Project", "project-a"),
            ("X-Memory-Project", "project-b"),
        ]
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_tools_call_request(str(uuid4())),
            headers=headers,
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert "must be provided exactly once" in result["content"][0]["text"]
    repository.get_by_id.assert_not_awaited()
