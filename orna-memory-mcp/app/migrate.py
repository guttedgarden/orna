"""Apply database migrations before starting the MCP runtime."""

import asyncio

from app.config import Settings, settings
from app.db import run_database_migrations


async def migrate_database(app_settings: Settings = settings) -> list[str]:
    """Apply pending migrations and emit a concise bootstrap result."""
    applied = await run_database_migrations(app_settings)
    if applied:
        print(f"Applied database migrations: {', '.join(applied)}")
    else:
        print("Database schema is up to date.")
    return applied


def main() -> None:
    """CLI entry point for the Compose migration service."""
    asyncio.run(migrate_database())


if __name__ == "__main__":
    main()
