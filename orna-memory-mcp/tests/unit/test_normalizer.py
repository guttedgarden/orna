import unicodedata

import pytest

from app.normalizer import (
    build_lexical_source,
    normalize_query_to_plain_tokens,
    split_identifier,
)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        (
            "ResponseProviderExecutor",
            ["ResponseProviderExecutor", "response", "provider", "executor"],
        ),
        ("memory_supersede", ["memory_supersede", "memory", "supersede"]),
        ("x-request-id", ["x-request-id", "x", "request", "id"]),
        (
            "src/core/Application.php",
            [
                "src/core/Application.php",
                "src",
                "core",
                "Application",
                "application",
                "php",
            ],
        ),
        (
            r"C:\src\App\Service.php",
            [r"C:\src\App\Service.php", "C", "c", "src", "App", "app", "Service", "service", "php"],
        ),
        ("HTTP2Client", ["HTTP2Client", "http2", "client"]),
        ("XMLHTTPParser", ["XMLHTTPParser", "xmlhttp", "parser"]),
        ("PostgreSQL16", ["PostgreSQL16", "postgre", "sql16"]),
        ("SHA-256", ["SHA-256", "SHA", "sha", "256"]),
        ("C++", ["C++", "C", "c"]),
        ("C#", ["C#", "C", "c"]),
        (".NET", [".NET", "NET", "net"]),
        ("МодульПамяти", ["МодульПамяти", "модуль", "памяти"]),
        (
            "обработчик_запросов",
            ["обработчик_запросов", "обработчик", "запросов"],
        ),
        ("тест-кейс", ["тест-кейс", "тест", "кейс"]),
    ],
)
def test_split_identifier(identifier: str, expected: list[str]) -> None:
    assert split_identifier(identifier) == expected


def test_split_identifier_deduplicates_empty_and_repeated_parts() -> None:
    assert split_identifier("foo__foo---bar") == ["foo__foo---bar", "foo", "bar"]
    assert split_identifier("   ") == []


def test_split_identifier_normalizes_unicode_to_nfc() -> None:
    nfc = "CaféParser"
    nfd = unicodedata.normalize("NFD", nfc)

    assert split_identifier(nfd) == split_identifier(nfc)


def test_build_lexical_source_is_deterministic_and_expands_identifiers() -> None:
    result = build_lexical_source(
        "Обработчик ответа",
        ["streaming", "ответы"],
        ["ResponseProviderExecutor", "src/core/Application.php"],
    )

    assert result == (
        "Обработчик ответа streaming ответы "
        "ResponseProviderExecutor response provider executor "
        "src/core/Application.php src core Application application php"
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("foo-bar", "foo bar"),
        ("x-request-id", "x request id"),
        ("foo & bar", "foo bar"),
        ('"quotes"', "quotes"),
        ("foo | !bar", "foo bar"),
        ("(foo)", "foo"),
        ("ResponseProviderExecutor", "responseproviderexecutor response provider executor"),
        (
            "src/core/Application.php",
            "src core application php",
        ),
        (
            "Как устроен ResponseProviderExecutor?",
            "как устроен responseproviderexecutor response provider executor",
        ),
        ("English и русский текст", "english и русский текст"),
        ("emoji🔎Identifier", "emoji identifier"),
        ("", ""),
        ("&&&", ""),
        ("   ", ""),
    ],
)
def test_normalize_query_to_plain_tokens(query: str, expected: str) -> None:
    result = normalize_query_to_plain_tokens(query)

    assert result == expected
    assert normalize_query_to_plain_tokens(result) == result


def test_normalize_query_normalizes_unicode_to_nfc() -> None:
    nfc = "CaféParser"
    nfd = unicodedata.normalize("NFD", nfc)

    assert normalize_query_to_plain_tokens(nfd) == normalize_query_to_plain_tokens(nfc)
