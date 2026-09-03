-- 0001_initial.sql: Initial schema for Orna Memory
-- Расширения
-- Примечание: пользователь БД для начальных миграций должен иметь права на создание расширения vector,
-- либо расширение должно быть предварительно создано в целевой базе данных.
CREATE EXTENSION IF NOT EXISTS vector;

-- Основная таблица памяти
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY,
    logical_id UUID NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    -- previous_revision вычисляется автоматически для строгой проверки линейности цепочки ревизий
    previous_revision INTEGER GENERATED ALWAYS AS (revision - 1) STORED,
    supersedes_id UUID NULL,
    scope TEXT NOT NULL CHECK (scope IN ('global', 'project')),
    project_id TEXT NULL,
    memory_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'archived')),
    content TEXT NOT NULL,
    -- SHA-256 хэш контента должен составлять ровно 32 байта
    content_hash BYTEA NOT NULL CHECK (octet_length(content_hash) = 32),
    tags TEXT[] NOT NULL DEFAULT '{}',
    identifiers TEXT[] NOT NULL DEFAULT '{}',
    lexical_source TEXT NOT NULL,
    lexical_profile_version TEXT NOT NULL,
    -- Автоматически генерируемый tsvector с конфигурацией simple (без стемминга)
    lexical_text TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', lexical_source)) STORED,
    embedding VECTOR(1024) NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_profile_version TEXT NOT NULL,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status_changed_at TIMESTAMPTZ NULL,

    -- Инварианты целостности
    -- Изоляция scope: global не имеет project_id, project обязательно имеет project_id
    CONSTRAINT chk_memories_scope_project
        CHECK ((scope = 'global' AND project_id IS NULL) OR (scope = 'project' AND project_id IS NOT NULL)),

    -- Первая ревизия не имеет родителя, последующие обязательно ссылаются на supersedes_id
    CONSTRAINT chk_memories_revision_supersedes
        CHECK ((revision = 1 AND supersedes_id IS NULL) OR (revision > 1 AND supersedes_id IS NOT NULL)),

    -- Запрет ссылки на самого себя
    CONSTRAINT chk_memories_supersedes_not_self
        CHECK (supersedes_id IS NULL OR supersedes_id <> id),

    -- Уникальность номера ревизии внутри одной логической сущности
    CONSTRAINT uq_memories_logical_revision
        UNIQUE (logical_id, revision),

    -- Уникальный кортеж для ссылки внешнего ключа
    CONSTRAINT uq_memories_id_logical_revision
        UNIQUE (id, logical_id, revision),

    -- Внешний ключ, гарантирующий ссылку строго на предыдущую ревизию той же сущности
    CONSTRAINT fk_memories_supersedes
        FOREIGN KEY (supersedes_id, logical_id, previous_revision)
        REFERENCES memories (id, logical_id, revision)
);

-- Индексы
-- Ровно одна активная запись на каждый logical_id
CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_active_logical
    ON memories (logical_id)
    WHERE status = 'active';

-- Предотвращение ветвления: одна запись не может быть заменена дважды
CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_supersedes
    ON memories (supersedes_id)
    WHERE supersedes_id IS NOT NULL;

-- Полнотекстовый поиск GIN
CREATE INDEX IF NOT EXISTS idx_memories_lexical_text
    ON memories
    USING gin (lexical_text);

-- Векторный индекс HNSW по косинусному расстоянию для активных записей
CREATE INDEX IF NOT EXISTS idx_memories_embedding_active
    ON memories
    USING hnsw (embedding vector_cosine_ops)
    WHERE status = 'active';

-- Быстрый поиск дубликатов по хэшу контента
CREATE INDEX IF NOT EXISTS idx_memories_active_content_hash
    ON memories (scope, project_id, content_hash)
    WHERE status = 'active';

-- Триггер иммутабельности ревизий памяти
-- Существующие ревизии памяти неизменяемы; разрешено менять только статус жизненного цикла
CREATE OR REPLACE FUNCTION prevent_memory_revision_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF
        NEW.id IS DISTINCT FROM OLD.id
        OR NEW.logical_id IS DISTINCT FROM OLD.logical_id
        OR NEW.revision IS DISTINCT FROM OLD.revision
        OR NEW.supersedes_id IS DISTINCT FROM OLD.supersedes_id
        OR NEW.scope IS DISTINCT FROM OLD.scope
        OR NEW.project_id IS DISTINCT FROM OLD.project_id
        OR NEW.memory_type IS DISTINCT FROM OLD.memory_type
        OR NEW.content IS DISTINCT FROM OLD.content
        OR NEW.content_hash IS DISTINCT FROM OLD.content_hash
        OR NEW.tags IS DISTINCT FROM OLD.tags
        OR NEW.identifiers IS DISTINCT FROM OLD.identifiers
        OR NEW.lexical_source IS DISTINCT FROM OLD.lexical_source
        OR NEW.lexical_profile_version IS DISTINCT FROM OLD.lexical_profile_version
        OR NEW.embedding IS DISTINCT FROM OLD.embedding
        OR NEW.embedding_model IS DISTINCT FROM OLD.embedding_model
        OR NEW.embedding_profile_version IS DISTINCT FROM OLD.embedding_profile_version
        OR NEW.provenance IS DISTINCT FROM OLD.provenance
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'memory revisions are immutable; create a new revision instead';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_memories_immutable_revision ON memories;
CREATE TRIGGER trg_memories_immutable_revision
BEFORE UPDATE ON memories
FOR EACH ROW
EXECUTE FUNCTION prevent_memory_revision_mutation();

