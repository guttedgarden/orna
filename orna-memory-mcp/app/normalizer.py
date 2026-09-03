"""Детерминированная нормализация контента и lexical representation."""

import hashlib
import unicodedata
from itertools import pairwise


def _normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _remove_external_blank_lines(content: str) -> str:
    """Удаляет внешние пустые строки, сохраняя horizontal whitespace кода."""
    lines = content.split("\n")
    start = 0
    end = len(lines)

    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    return "\n".join(lines[start:end])


def canonical_content_hash(content: str) -> bytes:
    """Возвращает binary SHA-256 для канонического представления контента."""
    canonical = _normalize_unicode(content)
    canonical = canonical.replace("\r\n", "\n").replace("\r", "\n")
    canonical = _remove_external_blank_lines(canonical)
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _is_word_character(character: str) -> bool:
    # Combining marks считаются частью слова для decomposed Unicode edge cases.
    return character.isalnum() or unicodedata.category(character).startswith("M")


def _split_on_separators(value: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []

    for character in value:
        if _is_word_character(character):
            current.append(character)
        elif current:
            segments.append("".join(current))
            current = []

    if current:
        segments.append("".join(current))

    return segments


def _split_camel_case(value: str) -> list[str]:
    if not value:
        return []

    boundaries = [0]
    for index in range(1, len(value)):
        previous = value[index - 1]
        current = value[index]
        following = value[index + 1] if index + 1 < len(value) else ""

        lower_to_upper = previous.islower() and current.isupper()
        acronym_to_word = previous.isupper() and current.isupper() and following.islower()
        digit_to_word = previous.isdigit() and current.isupper() and following.islower()

        if lower_to_upper or acronym_to_word or digit_to_word:
            boundaries.append(index)

    boundaries.append(len(value))
    return [value[start:end] for start, end in pairwise(boundaries)]


def _append_unique(result: list[str], seen: set[str], token: str) -> None:
    if token and token not in seen:
        seen.add(token)
        result.append(token)


def split_identifier(identifier: str) -> list[str]:
    """Возвращает NFC-normalized identifier, его segments и lowercase parts."""
    original = _normalize_unicode(identifier).strip()
    if not original:
        return []

    result: list[str] = []
    seen: set[str] = set()
    _append_unique(result, seen, original)

    for segment in _split_on_separators(original):
        _append_unique(result, seen, segment)
        for component in _split_camel_case(segment):
            _append_unique(result, seen, component.lower())

    return result


def build_lexical_source(content: str, tags: list[str], identifiers: list[str]) -> str:
    """Собирает детерминированный source для PostgreSQL FTS ``simple``."""
    parts = [_normalize_unicode(content)]
    parts.extend(normalized for tag in tags if (normalized := _normalize_unicode(tag).strip()))
    for identifier in identifiers:
        parts.extend(split_identifier(identifier))

    return " ".join(part for part in parts if part)


def normalize_query_to_plain_tokens(query: str) -> str:
    """Разворачивает query identifiers в plain words для ``plainto_tsquery``."""
    normalized = _normalize_unicode(query)
    result: list[str] = []
    seen: set[str] = set()

    for segment in _split_on_separators(normalized):
        _append_unique(result, seen, segment.lower())
        for component in _split_camel_case(segment):
            _append_unique(result, seen, component.lower())

    return " ".join(result)
