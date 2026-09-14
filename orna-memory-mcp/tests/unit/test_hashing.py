import unicodedata

from app.normalizer import canonical_content_hash


def test_hash_is_binary_sha256_digest() -> None:
    digest = canonical_content_hash("content")

    assert isinstance(digest, bytes)
    assert len(digest) == 32


def test_hash_normalizes_line_endings() -> None:
    assert canonical_content_hash("first\r\nsecond\r") == canonical_content_hash("first\nsecond\n")


def test_hash_ignores_external_blank_lines() -> None:
    assert canonical_content_hash("\n\ncontent\n\n") == canonical_content_hash("content")
    assert canonical_content_hash(" \t\ncontent\n\t ") == canonical_content_hash("content")


def test_hash_normalizes_unicode_to_nfc() -> None:
    nfc = "Café"
    nfd = unicodedata.normalize("NFD", nfc)

    assert canonical_content_hash(nfd) == canonical_content_hash(nfc)


def test_hash_preserves_internal_code_whitespace() -> None:
    assert canonical_content_hash("foo($x)") != canonical_content_hash("foo( $x )")


def test_hash_preserves_horizontal_whitespace_on_non_empty_edge_lines() -> None:
    assert canonical_content_hash("return value") != canonical_content_hash("    return value")
    assert canonical_content_hash("value") != canonical_content_hash("value  ")
