"""Модуль работы с базой данных PostgreSQL и раннер миграций.

PostgreSQL database client, connection pool management, and serialized migration runner.
Phase 1 migrations are transactional-only. Support for non-transactional migrations
may be added when operational migrations such as CREATE INDEX CONCURRENTLY become necessary.

The database user used for initial migrations must have permission to install the `vector`
extension, or the extension must be provisioned in the target database beforehand.
"""

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from app.config import Settings

# Константа ключа advisory lock для сериализации миграций
MIGRATION_LOCK_KEY = 814729001

_MIGRATION_FILE_PATTERN = re.compile(r"^(\d{4})_.*\.sql$")
_INITIAL_MIGRATION_VERSION = "0001_initial.sql"
_INITIAL_BOOTSTRAP_RELATION_NAMES = (
    "memories",
    "uq_memories_active_logical",
    "uq_memories_supersedes",
    "idx_memories_lexical_text",
    "idx_memories_embedding_active",
    "idx_memories_active_content_hash",
)
_INITIAL_BOOTSTRAP_FUNCTION_NAMES = ("prevent_memory_revision_mutation",)


class MigrationError(Exception):
    """Базовое исключение для ошибок подсистемы миграций."""


class ChecksumMismatchError(MigrationError):
    """Хэш файла миграции на диске не совпадает с записью в базе данных."""


class DuplicateMigrationError(MigrationError):
    """Обнаружены файлы миграций с совпадающим номером версии."""


async def _assert_initial_bootstrap_has_no_owned_object_collisions(
    conn: asyncpg.Connection,
) -> None:
    """Отклоняет legacy bootstrap при неизвестных объектах, принадлежащих 0001."""
    relation_rows = await conn.fetch(
        """
        SELECT c.relname
        FROM pg_class AS c
        JOIN pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = ANY($1::text[])
        ORDER BY c.relname;
        """,
        _INITIAL_BOOTSTRAP_RELATION_NAMES,
    )
    function_rows = await conn.fetch(
        """
        SELECT p.proname
        FROM pg_proc AS p
        JOIN pg_namespace AS n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public'
          AND p.proname = ANY($1::text[])
        ORDER BY p.proname;
        """,
        _INITIAL_BOOTSTRAP_FUNCTION_NAMES,
    )

    conflicting_objects = [f"public.{row['relname']}" for row in relation_rows]
    conflicting_objects.extend(f"function public.{row['proname']}()" for row in function_rows)
    if conflicting_objects:
        objects = ", ".join(conflicting_objects)
        raise MigrationError(
            f"Initial migration '{_INITIAL_MIGRATION_VERSION}' is pending, but conflicting "
            f"Orna-owned object(s) already exist: {objects}. Verify or restore migration "
            "history before retrying; no schema was adopted or modified."
        )


async def init_connection(conn: asyncpg.Connection) -> None:
    """Инициализация соединения с регистрацией типа vector для asyncpg."""
    from pgvector.asyncpg import register_vector

    await register_vector(conn)


async def create_db_pool(settings: "Settings") -> asyncpg.Pool:
    """Создание пула соединений к PostgreSQL с регистрацией pgvector."""
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
        init=init_connection,
    )


async def _execute_migrations(
    conn: asyncpg.Connection,
    migrations_dir: Path,
) -> list[str]:
    """Внутренняя логика применения миграций на конкретном соединении."""
    if not migrations_dir.is_dir():
        raise MigrationError(f"Migrations directory does not exist: {migrations_dir}")

    # Обнаруживаем файлы NNNN_*.sql и проверяем уникальность префикса
    migration_files: list[Path] = []
    seen_prefixes: dict[str, str] = {}

    for file_path in migrations_dir.glob("*.sql"):
        match = _MIGRATION_FILE_PATTERN.match(file_path.name)
        if match:
            prefix = match.group(1)
            if prefix in seen_prefixes:
                raise DuplicateMigrationError(
                    f"Duplicate migration version prefix '{prefix}' found: "
                    f"'{seen_prefixes[prefix]}' and '{file_path.name}'"
                )
            seen_prefixes[prefix] = file_path.name
            migration_files.append(file_path)

    migration_files.sort(key=lambda p: p.name)

    # Транзакция с явным уровнем изоляции READ COMMITTED
    async with conn.transaction(isolation="read_committed"):
        # Блокировка advisory lock на время транзакции для предотвращения гонок
        await conn.execute(f"SELECT pg_advisory_xact_lock({MIGRATION_LOCK_KEY});")

        # Создание служебной таблицы версий миграций
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )

        # Читаем уже примененные миграции
        rows = await conn.fetch("SELECT version, checksum FROM schema_migrations ORDER BY version;")
        applied_migrations: dict[str, str] = {row["version"]: row["checksum"] for row in rows}

        # Проверяем контрольные суммы примененных миграций по байтам файлов на диске
        for version, stored_checksum in applied_migrations.items():
            matching_file = next((f for f in migration_files if f.name == version), None)
            if matching_file is None:
                raise MigrationError(
                    f"Applied migration file '{version}' not found in {migrations_dir}"
                )

            raw_bytes = matching_file.read_bytes()
            computed_checksum = hashlib.sha256(raw_bytes).hexdigest()
            if computed_checksum != stored_checksum:
                raise ChecksumMismatchError(
                    f"Checksum mismatch for migration '{version}': "
                    f"recorded {stored_checksum}, computed {computed_checksum}"
                )

        if _INITIAL_MIGRATION_VERSION not in applied_migrations and any(
            file_path.name == _INITIAL_MIGRATION_VERSION for file_path in migration_files
        ):
            await _assert_initial_bootstrap_has_no_owned_object_collisions(conn)

        # Применяем новые миграции
        applied_versions: list[str] = []
        for file_path in migration_files:
            if file_path.name in applied_migrations:
                continue

            raw_bytes = file_path.read_bytes()
            checksum = hashlib.sha256(raw_bytes).hexdigest()
            sql_content = raw_bytes.decode("utf-8")

            # Выполняем DDL скрипт миграции
            await conn.execute(sql_content)

            # Фиксируем успешное применение в schema_migrations
            await conn.execute(
                "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2);",
                file_path.name,
                checksum,
            )
            applied_versions.append(file_path.name)

    return applied_versions


async def run_migrations(
    pool_or_conn: asyncpg.Pool | asyncpg.Connection,
    migrations_dir: Path | str | None = None,
) -> list[str]:
    """Сериализованный раннер миграций схемы БД с проверкой контрольных сумм."""
    target_dir = (
        Path(migrations_dir)
        if migrations_dir is not None
        else Path(__file__).parent / "schema" / "migrations"
    )

    if isinstance(pool_or_conn, asyncpg.Pool):
        async with pool_or_conn.acquire() as conn:
            return await _execute_migrations(conn, target_dir)
    else:
        return await _execute_migrations(pool_or_conn, target_dir)


async def run_database_migrations(
    settings: "Settings",
    migrations_dir: Path | str | None = None,
) -> list[str]:
    """Запуск миграций через отдельное bootstrap-соединение до инициализации пула."""
    conn = await asyncpg.connect(settings.database_url)
    try:
        return await run_migrations(conn, migrations_dir=migrations_dir)
    finally:
        await conn.close()
