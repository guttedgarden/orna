import pytest

from app.config import Settings
from app.mcp_server import create_mcp_server


def _settings(token: str) -> Settings:
    return Settings(orna_memory_token=token, _env_file=None)


@pytest.mark.parametrize("configured_token", ["", " ", "\t\n"])
def test_server_composition_fails_fast_for_blank_token(configured_token):
    with pytest.raises(ValueError, match="configured bearer token must not be blank"):
        create_mcp_server(_settings(configured_token))
