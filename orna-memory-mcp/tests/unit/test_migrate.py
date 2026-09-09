from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.migrate import migrate_database


@pytest.mark.parametrize(
    ("applied", "expected_message"),
    [
        (["0001_initial.sql", "0002_reject_zero_embeddings.sql"], "Applied database migrations"),
        ([], "Database schema is up to date."),
    ],
)
async def test_migrate_database_reports_bootstrap_result(
    monkeypatch,
    capsys,
    applied,
    expected_message,
):
    config = Settings(postgres_password="test-password", _env_file=None)
    runner = AsyncMock(return_value=applied)
    monkeypatch.setattr("app.migrate.run_database_migrations", runner)

    result = await migrate_database(config)

    assert result == applied
    runner.assert_awaited_once_with(config)
    assert expected_message in capsys.readouterr().out
