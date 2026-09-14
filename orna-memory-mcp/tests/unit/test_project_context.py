import pytest
from starlette.datastructures import Headers

from app.project_context import ProjectHeaderError, resolve_project_header


@pytest.mark.parametrize(
    "project_id",
    [
        "a",
        "f5ai-backend",
        "Orna.Memory_2",
        "a" * 128,
    ],
)
def test_resolve_project_header_accepts_canonical_project_id(project_id):
    assert resolve_project_header({"X-Memory-Project": project_id}) == project_id


def test_resolve_project_header_name_is_case_insensitive():
    assert resolve_project_header({"x-memory-project": "orna-memory"}) == "orna-memory"


@pytest.mark.parametrize("headers", [None, {}, {"X-Other": "orna-memory"}])
def test_resolve_project_header_rejects_missing_header(headers):
    with pytest.raises(ProjectHeaderError, match="is required"):
        resolve_project_header(headers)


@pytest.mark.parametrize(
    "project_id",
    [
        "",
        " project",
        "project ",
        "project name",
        "-project",
        "project-",
        ".project",
        "project_",
        "project/other",
        "проект",
        "a" * 129,
    ],
)
def test_resolve_project_header_rejects_noncanonical_project_id(project_id):
    with pytest.raises(ProjectHeaderError, match="must be a canonical project id"):
        resolve_project_header({"X-Memory-Project": project_id})


def test_resolve_project_header_rejects_duplicate_header():
    headers = Headers(
        raw=[
            (b"x-memory-project", b"project-a"),
            (b"x-memory-project", b"project-b"),
        ]
    )

    with pytest.raises(ProjectHeaderError, match="must be provided exactly once"):
        resolve_project_header(headers)


def test_resolve_project_header_has_no_cross_request_state():
    first = resolve_project_header({"X-Memory-Project": "project-a"})
    second = resolve_project_header({"X-Memory-Project": "project-b"})

    assert first == "project-a"
    assert second == "project-b"
