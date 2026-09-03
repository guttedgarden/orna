"""Интеграционные тесты DDL-схемы и сериализованного раннера миграций.

Integration tests for DDL schema, pgvector operations, and serialized migration runner.
Tests run against PostgreSQL + pgvector in isolated temporary databases.
"""

import asyncio
import hashlib
import shutil
import uuid
from pathlib import Path

import asyncpg
import pytest

from app.config import Settings
from app.db import (
    ChecksumMismatchError,
    DuplicateMigrationError,
    MigrationError,
    create_db_pool,
    run_database_migrations,
    run_migrations,
)

# Хэш контента фиксированной длины 32 байта для тестов
DUMMY_HASH_32 = hashlib.sha256(b"dummy_content").digest()
DUMMY_VECTOR = [0.0] * 1024


@pytest.fixture
async def test_database():
    """Создает изолированную временную БД для теста и удаляет её после выполнения.
    """
    settings = Settings()
    db_name = f"orna_test_{uuid.uuid4().hex[:10]}"
    admin_conn = await asyncpg.connect(
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/template1"
    )
    await admin_conn.execute(f'CREATE DATABASE "{db_name}";')

    test_settings = settings.model_copy(
        update={
            "postgres_db": db_name,
            "database_url": (
                f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
                f"@{settings.postgres_host}:{settings.postgres_port}/{db_name}"
            ),
        }
    )

    try:
        yield test_settings
    finally:
        # Принудительно отключаем оставшиеся сессии перед удалением
        await admin_conn.execute(
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{db_name}' AND pid <> pg_backend_pid();"
        )
        await admin_conn.execute(f'DROP DATABASE "{db_name}";')
        await admin_conn.close()


@pytest.fixture
async def migrated_pool(test_database: Settings):
    """Инициализирует БД миграциями и возвращает подключенный пул соединений.
    """
    await run_database_migrations(test_database)
    pool = await create_db_pool(test_database)
    try:
        yield pool
    finally:
        await pool.close()


async def insert_memory(
    conn: asyncpg.Connection,
    *,
    id: uuid.UUID | None = None,
    logical_id: uuid.UUID | None = None,
    revision: int = 1,
    supersedes_id: uuid.UUID | None = None,
    scope: str = "global",
    project_id: str | None = None,
    memory_type: str = "fact",
    status: str = "active",
    content: str = "Sample content",
    content_hash: bytes = DUMMY_HASH_32,
    embedding: list[float] | None = None,
) -> uuid.UUID:
    """Вспомогательная функция вставки записи в таблицу memories."""
    record_id = id or uuid.uuid4()
    log_id = logical_id or uuid.uuid4()
    emb = embedding or DUMMY_VECTOR

    await conn.execute(
        """
        INSERT INTO memories (
            id, logical_id, revision, supersedes_id, scope, project_id,
            memory_type, status, content, content_hash, lexical_source,
            lexical_profile_version, embedding, embedding_model,
            embedding_profile_version
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
        );
        """,
        record_id,
        log_id,
        revision,
        supersedes_id,
        scope,
        project_id,
        memory_type,
        status,
        content,
        content_hash,
        content,
        "lexical-v1",
        emb,
        "intfloat/multilingual-e5-large",
        "e5-v1",
    )
    return record_id


class TestMigrationRunner:
    """Тесты раннера миграций: идемпотентность, транзакционность, блокировки."""

    async def test_run_migrations_initial(self, test_database: Settings):
        """Проверка первого запуска миграций: успешное применение 0001_initial.sql."""
        conn = await asyncpg.connect(test_database.database_url)
        try:
            applied = await run_migrations(conn)
            assert applied == ["0001_initial.sql"]

            # Проверяем запись в таблице schema_migrations
            rows = await conn.fetch("SELECT version, checksum FROM schema_migrations;")
            assert len(rows) == 1
            assert rows[0]["version"] == "0001_initial.sql"

            # Проверяем существование таблицы memories и расширения vector
            memories_exists = await conn.fetchval(
                "SELECT to_regclass('public.memories') IS NOT NULL;"
            )
            assert memories_exists is True
        finally:
            await conn.close()

    async def test_run_migrations_idempotent(self, test_database: Settings):
        """Повторный запуск миграций возвращает пустой список (0 новых миграций)."""
        conn = await asyncpg.connect(test_database.database_url)
        try:
            applied_first = await run_migrations(conn)
            assert applied_first == ["0001_initial.sql"]

            applied_second = await run_migrations(conn)
            assert applied_second == []
        finally:
            await conn.close()

    async def test_checksum_mismatch_detection(self, test_database: Settings, tmp_path: Path):
        """Обнаружение изменения содержимого уже примененного файла миграции."""
        # Копируем оригинальную миграцию во временный каталог
        src_dir = Path(__file__).parents[2] / "app" / "schema" / "migrations"
        shutil.copytree(src_dir, tmp_path / "migrations")
        mig_dir = tmp_path / "migrations"

        conn = await asyncpg.connect(test_database.database_url)
        try:
            # Применяем оригинальную миграцию
            applied = await run_migrations(conn, migrations_dir=mig_dir)
            assert len(applied) == 1

            # Модифицируем файл миграции на диске
            initial_file = mig_dir / "0001_initial.sql"
            initial_file.write_text(
                initial_file.read_text(encoding="utf-8") + "\n-- modification\n",
                encoding="utf-8",
            )

            # Следующий запуск должен выбросить ChecksumMismatchError
            with pytest.raises(ChecksumMismatchError) as exc_info:
                await run_migrations(conn, migrations_dir=mig_dir)
            assert "Checksum mismatch" in str(exc_info.value)
        finally:
            await conn.close()

    async def test_applied_migration_missing_from_disk(
        self, test_database: Settings, tmp_path: Path
    ):
        """Проверка удаления файла уже примененной миграции (MigrationError)."""
        src_dir = Path(__file__).parents[2] / "app" / "schema" / "migrations"
        shutil.copytree(src_dir, tmp_path / "missing_migrations")
        mig_dir = tmp_path / "missing_migrations"

        conn = await asyncpg.connect(test_database.database_url)
        try:
            # Применяем миграцию
            applied = await run_migrations(conn, migrations_dir=mig_dir)
            assert len(applied) == 1

            # Удаляем примененный файл миграции с диска
            initial_file = mig_dir / "0001_initial.sql"
            initial_file.unlink()

            # Следующий запуск должен выбросить MigrationError
            with pytest.raises(MigrationError) as exc_info:
                await run_migrations(conn, migrations_dir=mig_dir)
            assert "not found" in str(exc_info.value)
        finally:
            await conn.close()

    async def test_duplicate_migration_prefix_detection(
        self, test_database: Settings, tmp_path: Path
    ):
        """Обнаружение дублирующихся номеров версий миграций (например, 0002_a и 0002_b)."""
        mig_dir = tmp_path / "duplicate_migrations"
        mig_dir.mkdir()
        (mig_dir / "0001_init.sql").write_text("SELECT 1;", encoding="utf-8")
        (mig_dir / "0002_foo.sql").write_text("SELECT 2;", encoding="utf-8")
        (mig_dir / "0002_bar.sql").write_text("SELECT 3;", encoding="utf-8")

        conn = await asyncpg.connect(test_database.database_url)
        try:
            with pytest.raises(DuplicateMigrationError) as exc_info:
                await run_migrations(conn, migrations_dir=mig_dir)
            assert "Duplicate migration version prefix '0002'" in str(exc_info.value)
        finally:
            await conn.close()

    async def test_migration_rollback_on_failure(self, test_database: Settings, tmp_path: Path):
        """Проверка отката при падении миграции (атомарность транзакции)."""
        mig_dir = tmp_path / "failing_migrations"
        mig_dir.mkdir()
        (mig_dir / "0001_first.sql").write_text(
            "CREATE TABLE temp_table (id INT);", encoding="utf-8"
        )
        (mig_dir / "0002_broken.sql").write_text("SYNTAX ERROR IN SQL;", encoding="utf-8")

        conn = await asyncpg.connect(test_database.database_url)
        try:
            with pytest.raises(asyncpg.PostgresError):
                await run_migrations(conn, migrations_dir=mig_dir)

            # Таблица temp_table не должна существовать из-за отката транзакции
            table_exists = await conn.fetchval(
                "SELECT to_regclass('public.temp_table') IS NOT NULL;"
            )
            assert table_exists is False

            # Таблица schema_migrations также должна быть откатана
            meta_exists = await conn.fetchval(
                "SELECT to_regclass('public.schema_migrations') IS NOT NULL;"
            )
            assert meta_exists is False
        finally:
            await conn.close()

    async def test_concurrent_migrations_advisory_lock(self, test_database: Settings):
        """Параллельный запуск run_migrations сериализуется advisory lock без ошибок."""
        # Создаем временный pool без init=register_vector до наката миграций
        pre_pool = await asyncpg.create_pool(
            dsn=test_database.database_url,
            min_size=2,
            max_size=4,
        )
        try:
            # Запускаем два раннера одновременно
            res1, res2 = await asyncio.gather(
                run_migrations(pre_pool),
                run_migrations(pre_pool),
            )

            # Ровно один применил миграцию, второй вернул пустой список
            results = [res1, res2]
            assert ["0001_initial.sql"] in results
            assert [] in results
        finally:
            await pre_pool.close()

    async def test_bootstrap_lifecycle_with_app_pool(self, test_database: Settings):
        """Полный жизненный цикл: bootstrap -> миграции -> пул приложения -> запись pgvector."""
        # 1. Bootstrap: миграции накатываются через прямое соединение
        applied = await run_database_migrations(test_database)
        assert applied == ["0001_initial.sql"]

        # 2. Создание пула приложения со строгим init_connection (register_vector)
        pool = await create_db_pool(test_database)
        try:
            async with pool.acquire() as conn:
                # Вставка и чтение вектора
                vec = [0.1] * 1024
                mid = await insert_memory(conn, embedding=vec)

                # Проверка чтения и вычисления cosine distance
                res = await conn.fetchrow(
                    "SELECT id, embedding <=> $1::vector AS distance FROM memories WHERE id = $2;",
                    vec,
                    mid,
                )
                assert res is not None
                assert abs(res["distance"]) < 1e-5
        finally:
            await pool.close()


class TestSchemaInvariants:
    """Тесты проверки инвариантов целостности схемы memories."""

    async def test_invariant_scope_project_check(self, migrated_pool: asyncpg.Pool):
        """Инвариант: scope='global' требует project_id IS NULL,
        scope='project' требует project_id NOT NULL.
        """
        async with migrated_pool.acquire() as conn:
            # Global с project_id -> ошибка
            with pytest.raises(asyncpg.CheckViolationError):
                await insert_memory(conn, scope="global", project_id="proj_1")

            # Project без project_id -> ошибка
            with pytest.raises(asyncpg.CheckViolationError):
                await insert_memory(conn, scope="project", project_id=None)

            # Корректные записи
            id_global = await insert_memory(conn, scope="global", project_id=None)
            id_project = await insert_memory(conn, scope="project", project_id="proj_1")
            assert id_global is not None
            assert id_project is not None

    async def test_invariant_revision_1_with_supersedes(self, migrated_pool: asyncpg.Pool):
        """Инвариант: revision = 1 не может иметь supersedes_id."""
        async with migrated_pool.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await insert_memory(conn, revision=1, supersedes_id=uuid.uuid4())

    async def test_invariant_revision_gt_1_without_supersedes(self, migrated_pool: asyncpg.Pool):
        """Инвариант: revision > 1 обязан иметь supersedes_id."""
        async with migrated_pool.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await insert_memory(conn, revision=2, supersedes_id=None)

    async def test_invariant_supersedes_not_self(self, migrated_pool: asyncpg.Pool):
        """Инвариант: supersedes_id не может ссылаться на саму себя."""
        async with migrated_pool.acquire() as conn:
            rec_id = uuid.uuid4()
            with pytest.raises(asyncpg.CheckViolationError):
                await insert_memory(conn, id=rec_id, revision=2, supersedes_id=rec_id)

    async def test_invariant_content_hash_length(self, migrated_pool: asyncpg.Pool):
        """Инвариант: content_hash обязан составлять ровно 32 байта."""
        async with migrated_pool.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await insert_memory(conn, content_hash=b"invalid_short_hash")

    async def test_invariant_unique_logical_revision(self, migrated_pool: asyncpg.Pool):
        """Инвариант: пара (logical_id, revision) уникальна."""
        async with migrated_pool.acquire() as conn:
            lid = uuid.uuid4()
            await insert_memory(conn, logical_id=lid, revision=1)

            # Попытка вставить вторую запись с тем же logical_id и revision=1
            with pytest.raises(asyncpg.UniqueViolationError):
                await insert_memory(conn, logical_id=lid, revision=1)

    async def test_invariant_unique_active_logical(self, migrated_pool: asyncpg.Pool):
        """Инвариант: не более одной активной записи на logical_id."""
        async with migrated_pool.acquire() as conn:
            lid = uuid.uuid4()
            id1 = await insert_memory(conn, logical_id=lid, revision=1, status="active")

            # Вставка revision=2 при активной revision=1
            with pytest.raises(asyncpg.UniqueViolationError):
                await insert_memory(
                    conn,
                    logical_id=lid,
                    revision=2,
                    supersedes_id=id1,
                    status="active",
                )

    async def test_invariant_unique_supersedes(self, migrated_pool: asyncpg.Pool):
        """Инвариант: предотвращение ветвления —
        две ревизии не могут ссылаться на одного родителя.
        """
        async with migrated_pool.acquire() as conn:
            lid = uuid.uuid4()
            id1 = await insert_memory(conn, logical_id=lid, revision=1, status="superseded")

            # Первая ревизия 2
            await insert_memory(
                conn,
                logical_id=lid,
                revision=2,
                supersedes_id=id1,
                status="superseded",
            )

            # Попытка создать параллельную ветку от того же id1
            with pytest.raises(asyncpg.UniqueViolationError):
                await insert_memory(
                    conn,
                    logical_id=lid,
                    revision=2,  # либо любая ревизия, ссылающаяся на id1
                    supersedes_id=id1,
                    status="active",
                )

    async def test_invariant_supersedes_foreign_logical_id(self, migrated_pool: asyncpg.Pool):
        """Инвариант: FK запрещает supersedes_id ссылаться на запись с другим logical_id."""
        async with migrated_pool.acquire() as conn:
            lid1 = uuid.uuid4()
            id1 = await insert_memory(conn, logical_id=lid1, revision=1, status="superseded")

            # Попытка сослаться на id1 из другого logical_id
            lid2 = uuid.uuid4()
            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await insert_memory(
                    conn,
                    logical_id=lid2,
                    revision=2,
                    supersedes_id=id1,
                    status="active",
                )

    async def test_invariant_supersedes_previous_revision(self, migrated_pool: asyncpg.Pool):
        """Инвариант P1: FK запрещает перескакивать через ревизии (например rev 3 -> rev 1)."""
        async with migrated_pool.acquire() as conn:
            lid = uuid.uuid4()
            id1 = await insert_memory(conn, logical_id=lid, revision=1, status="superseded")

            # Попытка вставить revision=3 со ссылкой на revision=1 (минуя revision=2)
            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await insert_memory(
                    conn,
                    logical_id=lid,
                    revision=3,
                    supersedes_id=id1,
                    status="active",
                )

    async def test_invariant_existing_revision_is_immutable(self, migrated_pool: asyncpg.Pool):
        """Инвариант: существующая ревизия памяти неизменяема (UPDATE данных запрещен)."""
        async with migrated_pool.acquire() as conn:
            mid = await insert_memory(conn, content="original content")

            # Попытка изменить content
            with pytest.raises(asyncpg.RaiseError) as exc_info:
                await conn.execute(
                    "UPDATE memories SET content = 'modified' WHERE id = $1;",
                    mid,
                )
            assert "memory revisions are immutable" in str(exc_info.value)

            # Попытка изменить embedding
            new_emb = [1.0] * 1024
            with pytest.raises(asyncpg.RaiseError) as exc_info:
                await conn.execute(
                    "UPDATE memories SET embedding = $1::vector WHERE id = $2;",
                    new_emb,
                    mid,
                )
            assert "memory revisions are immutable" in str(exc_info.value)

    async def test_invariant_lifecycle_fields_are_mutable(self, migrated_pool: asyncpg.Pool):
        """Инвариант: поля жизненного цикла (status, status_changed_at) разрешено обновлять."""
        async with migrated_pool.acquire() as conn:
            mid = await insert_memory(conn, status="active")

            # Обновление status и status_changed_at должно завершиться успешно
            await conn.execute(
                "UPDATE memories SET status = 'superseded', status_changed_at = NOW() "
                "WHERE id = $1;",
                mid,
            )
            row = await conn.fetchrow(
                "SELECT status, status_changed_at FROM memories WHERE id = $1;",
                mid,
            )
            assert row is not None
            assert row["status"] == "superseded"
            assert row["status_changed_at"] is not None
