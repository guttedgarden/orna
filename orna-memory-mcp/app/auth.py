"""Framework-neutral bearer credential verification for MCP transport composition."""

import hashlib
import hmac

from mcp.server.auth.provider import AccessToken, TokenVerifier

_LOCAL_CLIENT_ID = "orna-memory-local"


def _token_digest(token: str) -> bytes:
    """Возвращает digest фиксированной длины для timing-safe comparison."""
    return hashlib.sha256(token.encode("utf-8", errors="surrogatepass")).digest()


class StaticBearerTokenVerifier(TokenVerifier):
    """Проверяет единственный local-v1 bearer token по официальному MCP contract."""

    def __init__(self, configured_token: str) -> None:
        if not isinstance(configured_token, str) or not configured_token.strip():
            raise ValueError("configured bearer token must not be blank")
        self._configured_digest = _token_digest(configured_token)

    async def verify_token(self, token: str) -> AccessToken | None:
        """Возвращает MCP principal либо единообразно отклоняет credential."""
        if not isinstance(token, str):
            return None

        candidate_digest = _token_digest(token)
        if not hmac.compare_digest(candidate_digest, self._configured_digest):
            return None

        return AccessToken(token=token, client_id=_LOCAL_CLIENT_ID, scopes=[])
