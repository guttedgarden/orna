from datetime import UTC, datetime
from unittest.mock import AsyncMock, call
from uuid import UUID, uuid4

import pytest
from starlette.testclient import TestClient

from app.config import Settings
from app.mcp_server import INVALID_TOOL_ARGUMENTS_MESSAGE, create_http_app
from app.models import (
    EMBEDDING_DIMENSION,
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryStatus,
)
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService
from app.write import MemoryAddCommand, MemoryWriteService
from app.write_safety import PROBABLE_SECRET_MESSAGE, ProbableSecretError

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


def _memory_add_request(**arguments: object) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "memory_add",
            "arguments": arguments,
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


def _memory_search_request(**arguments: object) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "memory_search",
            "arguments": arguments,
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
    tool_name: str = "memory_get",
) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "MCP-Protocol-Version": _PROTOCOL_VERSION,
        "Mcp-Method": method,
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if method == "tools/call":
        headers["Mcp-Name"] = tool_name
    if project_id is not None:
        headers["X-Memory-Project"] = project_id
    return headers


def _settings(token: str) -> Settings:
    return Settings(orna_memory_token=token, _env_file=None)


def _repository() -> AsyncMock:
    return AsyncMock(spec=MemoryRepository)


def _write_service() -> AsyncMock:
    return AsyncMock(spec=MemoryWriteService)


def _search_service() -> AsyncMock:
    return AsyncMock(spec=MemorySearchService)


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


def _search_result(record: MemoryRecord) -> MemorySearchResult:
    return MemorySearchResult(
        id=record.id,
        logical_id=record.logical_id,
        revision=record.revision,
        scope=record.scope,
        project_id=record.project_id,
        memory_type=record.memory_type,
        status=record.status,
        content=record.content,
        tags=list(record.tags),
        identifiers=list(record.identifiers),
        provenance=dict(record.provenance),
        rrf_score=0.031,
        rank_dense=1,
        rank_lexical=2,
        distance=0.12,
        lexical_score=0.5,
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
    assert [tool["name"] for tool in tools] == [
        "memory_get",
        "memory_add",
        "memory_search",
    ]
    memory_get, memory_add, memory_search = tools
    assert memory_get["inputSchema"]["additionalProperties"] is False
    assert memory_get["inputSchema"]["required"] == ["memory_id"]
    assert memory_get["inputSchema"]["properties"]["memory_id"]["format"] == "uuid"
    assert memory_get["annotations"] == {
        "readOnlyHint": True,
        "openWorldHint": False,
    }
    assert "embedding" not in memory_get["outputSchema"]["properties"]
    assert "lexical_source" not in memory_get["outputSchema"]["properties"]
    assert "content_hash" not in memory_get["outputSchema"]["properties"]
    assert memory_add["inputSchema"]["additionalProperties"] is False
    assert set(memory_add["inputSchema"]["properties"]) == {
        "content",
        "memory_type",
        "tags",
        "identifiers",
    }
    assert memory_add["inputSchema"]["required"] == ["content", "memory_type"]
    assert memory_add["inputSchema"]["properties"]["tags"]["default"] == []
    assert memory_add["inputSchema"]["properties"]["identifiers"]["default"] == []
    assert memory_add["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
    assert "durable" in memory_add["description"]
    assert "always project-scoped" in memory_add["description"]
    assert "Credentials are forbidden" in memory_add["description"]
    assert memory_search["inputSchema"]["additionalProperties"] is False
    assert set(memory_search["inputSchema"]["properties"]) == {"query", "memory_type"}
    assert memory_search["inputSchema"]["required"] == ["query"]
    assert memory_search["inputSchema"]["properties"]["memory_type"]["default"] is None
    assert memory_search["annotations"] == {
        "readOnlyHint": True,
        "openWorldHint": False,
    }
    assert memory_search["outputSchema"]["required"] == ["results"]
    assert set(memory_search["outputSchema"]["properties"]) == {"results"}
    output_schema = str(memory_search["outputSchema"])
    for internal_field in (
        "embedding",
        "content_hash",
        "lexical_source",
        "rrf_score",
        "rank_dense",
        "rank_lexical",
        "distance",
        "lexical_score",
    ):
        assert internal_field not in output_schema
    assert "durable project experience" in memory_search["description"]
    assert "applicable global memories" in memory_search["description"]
    assert "memory_type" in memory_search["description"]
    assert "at most 5" in memory_search["description"]
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
    assert result["content"][0]["text"] == INVALID_TOOL_ARGUMENTS_MESSAGE
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
    assert result["content"][0]["text"] == INVALID_TOOL_ARGUMENTS_MESSAGE
    assert "not-a-uuid" not in result["content"][0]["text"]
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


def test_memory_add_is_project_scoped_with_server_owned_provenance_and_db_timestamp():
    repository = _repository()
    write_service = _write_service()
    record = _memory_record().model_copy(
        update={
            "revision": 1,
            "supersedes_id": None,
            "status": MemoryStatus.ACTIVE,
            "content": "Use PostgreSQL for migration tests.",
            "tags": ["database"],
            "identifiers": ["MigrationRunner"],
            "provenance": {
                "created_by": "codex",
                "source": {"kind": "agent_explicit_add"},
                "project_id": "project-a",
            },
            "status_changed_at": None,
        }
    )
    write_service.add.return_value = record
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        write_service=write_service,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_add_request(
                content=record.content,
                memory_type="decision",
                tags=record.tags,
                identifiers=record.identifiers,
            ),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_add",
            ),
        )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["created_at"] == "2026-09-08T12:00:00Z"
    assert result["structuredContent"]["provenance"] == record.provenance
    assert "created_at" not in result["structuredContent"]["provenance"]
    for internal_field in (
        "embedding",
        "content_hash",
        "lexical_source",
        "embedding_model",
        "embedding_profile_version",
        "lexical_profile_version",
    ):
        assert internal_field not in result["structuredContent"]
    assert "mcp-session-id" not in response.headers

    write_service.add.assert_awaited_once()
    (command,) = write_service.add.await_args.args
    assert isinstance(command, MemoryAddCommand)
    assert command.scope is MemoryScope.PROJECT
    assert command.provenance == record.provenance
    assert write_service.add.await_args.kwargs == {"project_id": "project-a"}


@pytest.mark.parametrize(
    ("project_id", "message"),
    [
        (None, "X-Memory-Project header is required"),
        ("invalid project", "X-Memory-Project header must be a canonical project id"),
    ],
)
def test_memory_add_rejects_invalid_project_header_before_write(project_id, message):
    repository = _repository()
    write_service = _write_service()
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        write_service=write_service,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_add_request(content="Safe memory", memory_type="decision"),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id=project_id,
                tool_name="memory_add",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert message in result["content"][0]["text"]
    write_service.add.assert_not_awaited()
    assert repository.mock_calls == []


def test_memory_add_preserves_handler_tool_error_for_invalid_domain_arguments():
    repository = _repository()
    write_service = _write_service()
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        write_service=write_service,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_add_request(content="", memory_type="decision"),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_add",
            ),
        )

    result = response.json()["result"]
    error_text = result["content"][0]["text"]
    assert result["isError"] is True
    assert error_text.endswith("invalid memory arguments")
    assert error_text != INVALID_TOOL_ARGUMENTS_MESSAGE
    write_service.add.assert_not_awaited()
    assert repository.mock_calls == []


def test_memory_add_safety_error_is_publicly_safe():
    repository = _repository()
    write_service = _write_service()
    rejected = "password=example-value"
    write_service.add.side_effect = ProbableSecretError(PROBABLE_SECRET_MESSAGE)
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        write_service=write_service,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_add_request(content=rejected, memory_type="decision"),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_add",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert PROBABLE_SECRET_MESSAGE in result["content"][0]["text"]
    assert rejected not in result["content"][0]["text"]
    assert repository.mock_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scope", "global"),
        ("project_id", "other"),
        ("created_at", "2026-09-08T12:00:00Z"),
        ("provenance", {"api_key": "synthetic-secret-value"}),
        ("embedding", [1.0]),
    ],
)
def test_memory_add_rejects_forbidden_extra_arguments_before_write(field, value):
    repository = _repository()
    write_service = _write_service()
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        write_service=write_service,
    )
    arguments = {
        "content": "Use PostgreSQL for migration tests.",
        "memory_type": "decision",
        field: value,
    }

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_add_request(**arguments),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_add",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == INVALID_TOOL_ARGUMENTS_MESSAGE
    assert str(value) not in result["content"][0]["text"]
    write_service.add.assert_not_awaited()
    assert repository.mock_calls == []


def test_memory_add_rejects_invalid_allowed_argument_without_reflecting_sensitive_value():
    repository = _repository()
    write_service = _write_service()
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        write_service=write_service,
    )
    sensitive_value = "synthetic-sensitive-value"

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_add_request(
                content="Use PostgreSQL for migration tests.",
                memory_type="decision",
                tags={"api_key": sensitive_value},
            ),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_add",
            ),
        )

    result = response.json()["result"]
    error_text = result["content"][0]["text"]
    assert result["isError"] is True
    assert error_text == INVALID_TOOL_ARGUMENTS_MESSAGE
    assert sensitive_value not in error_text
    assert "input_value" not in error_text
    assert "input_type" not in error_text
    assert "errors.pydantic.dev" not in error_text
    write_service.add.assert_not_awaited()
    assert repository.mock_calls == []


def test_memory_search_uses_project_context_and_returns_public_wrapper():
    repository = _repository()
    search_service = _search_service()
    project_record = _memory_record().model_copy(update={"status": MemoryStatus.ACTIVE})
    global_record = project_record.model_copy(
        update={
            "id": UUID("019cff03-d6db-7772-89b8-e18dc19a9040"),
            "logical_id": UUID("019cff03-d6db-7772-89b8-e18dc19a9041"),
            "scope": MemoryScope.GLOBAL,
            "project_id": None,
            "content": "Keep migration logs for every project.",
        }
    )
    search_service.search.return_value = [
        _search_result(project_record),
        _search_result(global_record),
    ]
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        search_service=search_service,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_search_request(query="migration database", memory_type="decision"),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_search",
            ),
        )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"] == {
        "results": [
            {
                "id": str(project_record.id),
                "logical_id": str(project_record.logical_id),
                "revision": project_record.revision,
                "scope": "project",
                "project_id": "project-a",
                "memory_type": "decision",
                "status": "active",
                "content": project_record.content,
                "tags": project_record.tags,
                "identifiers": project_record.identifiers,
                "provenance": project_record.provenance,
            },
            {
                "id": str(global_record.id),
                "logical_id": str(global_record.logical_id),
                "revision": global_record.revision,
                "scope": "global",
                "project_id": None,
                "memory_type": "decision",
                "status": "active",
                "content": global_record.content,
                "tags": global_record.tags,
                "identifiers": global_record.identifiers,
                "provenance": global_record.provenance,
            },
        ]
    }
    for internal_field in (
        "embedding",
        "content_hash",
        "lexical_source",
        "rrf_score",
        "rank_dense",
        "rank_lexical",
        "distance",
        "lexical_score",
    ):
        assert internal_field not in result["structuredContent"]["results"][0]
    search_service.search.assert_awaited_once()
    (search_query,) = search_service.search.await_args.args
    assert search_query == MemorySearchQuery(
        query="migration database",
        memory_type="decision",
        limit=5,
    )
    assert search_service.search.await_args.kwargs == {"project_id": "project-a"}
    assert repository.mock_calls == []
    assert "mcp-session-id" not in response.headers


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limit", 100),
        ("project_id", "other"),
        ("scope", "global"),
        ("rrf_k", 1),
        ("unknown", True),
    ],
)
def test_memory_search_rejects_forbidden_extra_arguments_before_search(field, value):
    repository = _repository()
    search_service = _search_service()
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        search_service=search_service,
    )
    arguments = {"query": "migration database", field: value}

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_search_request(**arguments),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_search",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == INVALID_TOOL_ARGUMENTS_MESSAGE
    assert str(value) not in result["content"][0]["text"]
    search_service.search.assert_not_awaited()
    assert repository.mock_calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "migration database", "memory_type": ""},
        {"query": "migration database", "memory_type": " convention"},
    ],
)
def test_memory_search_rejects_blank_query_and_invalid_memory_type(arguments):
    repository = _repository()
    search_service = _search_service()
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        search_service=search_service,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_search_request(**arguments),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_search",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"].endswith("invalid memory search arguments")
    search_service.search.assert_not_awaited()
    assert repository.mock_calls == []


@pytest.mark.parametrize(
    ("project_id", "message"),
    [
        (None, "X-Memory-Project header is required"),
        ("invalid project", "X-Memory-Project header must be a canonical project id"),
    ],
)
def test_memory_search_rejects_invalid_project_header_before_search(project_id, message):
    repository = _repository()
    search_service = _search_service()
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        search_service=search_service,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_search_request(query="migration database"),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id=project_id,
                tool_name="memory_search",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert message in result["content"][0]["text"]
    search_service.search.assert_not_awaited()
    assert repository.mock_calls == []


def test_memory_search_rejects_duplicate_project_header_before_search():
    repository = _repository()
    search_service = _search_service()
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        search_service=search_service,
    )
    headers = list(
        _mcp_headers(
            "correct-token",
            method="tools/call",
            tool_name="memory_search",
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
            json=_memory_search_request(query="migration database"),
            headers=headers,
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert "must be provided exactly once" in result["content"][0]["text"]
    search_service.search.assert_not_awaited()
    assert repository.mock_calls == []


def test_memory_search_unexpected_failure_does_not_expose_internal_diagnostics():
    repository = _repository()
    search_service = _search_service()
    internal_diagnostic = "SELECT embedding FROM memories password=database-secret"
    search_service.search.side_effect = RuntimeError(internal_diagnostic)
    app = create_http_app(
        _settings("correct-token"),
        repository=repository,
        search_service=search_service,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/mcp",
            json=_memory_search_request(query="migration database"),
            headers=_mcp_headers(
                "correct-token",
                method="tools/call",
                project_id="project-a",
                tool_name="memory_search",
            ),
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert internal_diagnostic not in result["content"][0]["text"]
    assert "SELECT" not in result["content"][0]["text"]
