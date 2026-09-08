import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastembed.common.preprocessor_utils import load_tokenizer

from app.config import Settings
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE
from app.embeddings import prepare_memory_text
from app.write_safety import (
    MEMORY_TOO_LONG_MESSAGE,
    PROBABLE_SECRET_MESSAGE,
    E5LengthGuard,
    MemoryTooLongError,
    MemoryWriteSafety,
    ProbableSecretError,
    SecretScanner,
)


def _real_cache_dir() -> Path:
    configured = os.environ.get("ORNA_TEST_E5_CACHE_DIR")
    cache_dir = Path(configured) if configured else Settings(_env_file=None).embedding_cache_dir
    snapshot = ACTIVE_EMBEDDING_PROFILE.snapshot_path(cache_dir)
    if not snapshot.is_dir():
        pytest.skip(
            "pinned E5 tokenizer cache unavailable; run the documented model-cache workflow "
            "and set ORNA_TEST_E5_CACHE_DIR"
        )
    return cache_dir


@pytest.fixture(scope="module")
def real_tokenizer():
    snapshot = ACTIVE_EMBEDDING_PROFILE.snapshot_path(_real_cache_dir())
    tokenizer, _special_tokens = load_tokenizer(snapshot)
    tokenizer.no_truncation()
    return tokenizer


@pytest.fixture(scope="module")
def real_guard() -> E5LengthGuard:
    return E5LengthGuard(Settings(embedding_cache_dir=_real_cache_dir(), _env_file=None))


def _content_for_prepared_token_count(tokenizer, target: int) -> str:
    """Строит boundary input по фактическому output pinned tokenizer."""
    content = "a"
    for _index in range(target * 3):
        actual = len(tokenizer.encode(prepare_memory_text(content)).ids)
        if actual == target:
            return content
        if actual > target:
            break
        content += " a"
    raise AssertionError(f"could not construct content with exactly {target} input IDs")


def test_real_tokenizer_accepts_value_below_boundary(real_tokenizer, real_guard):
    content = _content_for_prepared_token_count(real_tokenizer, 511)

    real_guard.validate(prepare_memory_text(content))


def test_real_tokenizer_accepts_exactly_512_input_ids(real_tokenizer, real_guard):
    content = _content_for_prepared_token_count(real_tokenizer, 512)

    assert len(real_tokenizer.encode(prepare_memory_text(content)).ids) == 512
    real_guard.validate(prepare_memory_text(content))


def test_real_tokenizer_rejects_513_input_ids(real_tokenizer, real_guard):
    content = _content_for_prepared_token_count(real_tokenizer, 513)

    assert len(real_tokenizer.encode(prepare_memory_text(content)).ids) == 513
    with pytest.raises(MemoryTooLongError) as error:
        real_guard.validate(prepare_memory_text(content))

    assert str(error.value) == MEMORY_TOO_LONG_MESSAGE
    assert content not in str(error.value)


def test_boundary_includes_special_tokens_and_canonical_passage_prefix(
    real_tokenizer,
    real_guard,
):
    content = _content_for_prepared_token_count(real_tokenizer, 513)
    prepared = prepare_memory_text(content)

    assert len(real_tokenizer.encode(content).ids) < len(real_tokenizer.encode(prepared).ids)
    assert len(real_tokenizer.encode(content).ids) <= 512
    with pytest.raises(MemoryTooLongError):
        real_guard.validate(prepared)


def test_existing_passage_prefix_uses_same_inference_sequence(real_tokenizer, real_guard):
    content = _content_for_prepared_token_count(real_tokenizer, 512)
    plain_prepared = prepare_memory_text(content)
    prefixed_prepared = prepare_memory_text(f"passage: {content}")

    assert plain_prepared == prefixed_prepared
    assert real_tokenizer.encode(plain_prepared).ids == real_tokenizer.encode(prefixed_prepared).ids
    real_guard.validate(prefixed_prepared)


def test_guard_does_not_mutate_separate_runtime_tokenizer(real_tokenizer):
    snapshot = ACTIVE_EMBEDDING_PROFILE.snapshot_path(_real_cache_dir())
    runtime_tokenizer, _special_tokens = load_tokenizer(snapshot)
    runtime_truncation = dict(runtime_tokenizer.truncation or {})
    runtime_padding = dict(runtime_tokenizer.padding or {})
    guard = E5LengthGuard(Settings(embedding_cache_dir=_real_cache_dir(), _env_file=None))

    guard.validate(prepare_memory_text("A short durable claim."))

    assert runtime_tokenizer.truncation == runtime_truncation
    assert runtime_tokenizer.padding == runtime_padding
    assert real_tokenizer.truncation is None


@pytest.mark.parametrize(
    "value",
    [
        "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----",
        "Bearer exampleCredential1234",
        "Authorization: Bearer exampleCredential1234",
        "abcd_efgh.ijkl-mnop.qrst_uvwx",
        "sk-ExampleCredential123456",
        "ghp_ExampleCredential1234567890",
        "glpat-ExampleCredential1234567890",
        "password=example-value",
        "PassWd: example-value",
        "API_KEY = example-value",
        "api-key: example-value",
    ],
)
def test_secret_scanner_rejects_phase_one_patterns_without_reflection(value):
    scanner = SecretScanner()

    with pytest.raises(ProbableSecretError) as error:
        scanner.validate([value])

    assert str(error.value) == PROBABLE_SECRET_MESSAGE
    assert value not in str(error.value)


@pytest.mark.parametrize(
    "value",
    [
        "019cff03-d6db-7772-89b8-e18dc19a9038",
        "4f3c2a1b0e9d8c7b6a5f4e3d2c1b0a9988776655",
        "ResponseProviderExecutor",
        "sk-learn handles feature preprocessing",
        "Bearer authentication uses an Authorization header.",
        "Bearer token handling must use constant-time comparison.",
        "The terms password and api key are discussed without assigned values.",
        "Используй PostgreSQL для миграционных тестов.",
        "Keep the repository boundary deterministic and explicit.",
    ],
)
def test_secret_scanner_accepts_expected_non_secrets(value):
    SecretScanner().validate([value])


@pytest.mark.parametrize("field", ["content", "memory_type", "tag", "identifier"])
def test_write_safety_scans_every_agent_controlled_persisted_string(field, monkeypatch):
    rejected = "password=example-value"
    values = {
        "content": "Safe content",
        "memory_type": "decision",
        "tags": ["database"],
        "identifiers": ["MigrationRunner"],
    }
    if field == "content":
        values["content"] = rejected
    elif field == "memory_type":
        values["memory_type"] = rejected
    elif field == "tag":
        values["tags"] = [rejected]
    else:
        values["identifiers"] = [rejected]

    length_guard = MagicMock()
    prepare = MagicMock(return_value="passage: should not be built")
    monkeypatch.setattr("app.write_safety.prepare_memory_text", prepare)
    safety = MemoryWriteSafety(length_guard)

    with pytest.raises(ProbableSecretError) as error:
        safety.validate(
            **values,
        )

    assert rejected not in str(error.value)
    prepare.assert_not_called()
    length_guard.validate.assert_not_called()


def test_valid_memory_reaches_length_guard_after_secret_scan():
    length_guard = MagicMock()
    safety = MemoryWriteSafety(length_guard)
    prepared = prepare_memory_text("Use PostgreSQL for migration tests.")

    safety.validate(
        content="Use PostgreSQL for migration tests.",
        memory_type="decision",
        tags=["database"],
        identifiers=["MigrationRunner"],
    )

    length_guard.validate.assert_called_once_with(prepared)
