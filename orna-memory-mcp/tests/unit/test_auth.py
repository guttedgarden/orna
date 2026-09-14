import hmac

import pytest
from mcp.server.auth.provider import AccessToken, TokenVerifier

from app.auth import StaticBearerTokenVerifier


async def test_valid_token_returns_minimal_mcp_access_token():
    verifier: TokenVerifier = StaticBearerTokenVerifier("correct-token")

    result = await verifier.verify_token("correct-token")

    assert isinstance(result, AccessToken)
    assert result.token == "correct-token"
    assert result.client_id == "orna-memory-local"
    assert result.scopes == []
    assert result.expires_at is None
    assert result.resource is None
    assert result.subject is None
    assert result.claims is None


@pytest.mark.parametrize(
    "candidate",
    [
        "wrong-token",
        "correct-token-extra",
        "токен",
        "\ud800",
        "x" * 100_000,
    ],
)
async def test_invalid_tokens_of_any_length_and_unicode_are_rejected(candidate):
    verifier = StaticBearerTokenVerifier("correct-token")

    assert await verifier.verify_token(candidate) is None


async def test_empty_candidate_token_is_rejected():
    verifier = StaticBearerTokenVerifier("correct-token")

    assert await verifier.verify_token("") is None


@pytest.mark.parametrize("configured_token", ["", " ", "\t\n"])
def test_blank_configured_token_fails_fast_without_exposing_value(configured_token):
    with pytest.raises(ValueError) as exc_info:
        StaticBearerTokenVerifier(configured_token)

    assert str(exc_info.value) == "configured bearer token must not be blank"
    assert repr(configured_token) not in str(exc_info.value)


async def test_invalid_token_is_compared_in_fixed_digest_form(monkeypatch):
    compared_values: list[tuple[bytes, bytes]] = []
    real_compare_digest = hmac.compare_digest

    def recording_compare_digest(left: bytes, right: bytes) -> bool:
        compared_values.append((left, right))
        return real_compare_digest(left, right)

    monkeypatch.setattr("app.auth.hmac.compare_digest", recording_compare_digest)
    verifier = StaticBearerTokenVerifier("configured-secret")

    assert await verifier.verify_token("short") is None
    assert len(compared_values) == 1
    assert len(compared_values[0][0]) == 32
    assert len(compared_values[0][1]) == 32


async def test_rejected_token_does_not_expose_credentials(caplog):
    configured_token = "configured-secret"
    candidate_token = "presented-secret"
    verifier = StaticBearerTokenVerifier(configured_token)

    result = await verifier.verify_token(candidate_token)

    assert result is None
    assert configured_token not in caplog.text
    assert candidate_token not in caplog.text
