"""Request-scoped project context resolution for MCP calls."""

import re
from collections.abc import Mapping

PROJECT_HEADER_NAME = "X-Memory-Project"
_PROJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?")


class ProjectHeaderError(ValueError):
    """HTTP request не содержит однозначный канонический project context."""


def _header_values(headers: Mapping[str, str], name: str) -> list[object]:
    """Читает все значения, если transport mapping сохраняет duplicate headers."""
    getlist = getattr(headers, "getlist", None)
    if callable(getlist):
        return list(getlist(name))

    normalized_name = name.lower()
    return [value for key, value in headers.items() if key.lower() == normalized_name]


def resolve_project_header(headers: Mapping[str, str] | None) -> str:
    """Возвращает project ID только из одного валидного request header.

    Resolver намеренно не нормализует значение и не хранит его между вызовами:
    точный project ID вычисляется заново для каждого MCP request context.
    """
    if headers is None:
        raise ProjectHeaderError(f"{PROJECT_HEADER_NAME} header is required")

    values = _header_values(headers, PROJECT_HEADER_NAME)
    if not values:
        raise ProjectHeaderError(f"{PROJECT_HEADER_NAME} header is required")
    if len(values) != 1:
        raise ProjectHeaderError(f"{PROJECT_HEADER_NAME} header must be provided exactly once")

    project_id = values[0]
    if not isinstance(project_id, str) or _PROJECT_ID_PATTERN.fullmatch(project_id) is None:
        raise ProjectHeaderError(f"{PROJECT_HEADER_NAME} header must be a canonical project id")
    return project_id
