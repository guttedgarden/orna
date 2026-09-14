"""PostgreSQL repository for memory persistence and retrieval."""

import json
from collections.abc import Mapping
from typing import Any, Literal
from uuid import UUID

import asyncpg
from pgvector import Vector

from app.config import Settings
from app.models import MemoryInsertRecord, MemoryRecord
from app.normalizer import LexicalQueryGroup

DenseSearchStrategy = Literal["exact", "hnsw"]


def _build_lexical_query_expression(
    lexical_query_groups: list[LexicalQueryGroup],
) -> tuple[str, list[str]]:
    """Собирает fixed SQL expression и bound values для групп альтернатив."""
    expressions: list[str] = []
    parameters: list[str] = []

    for raw, expanded in lexical_query_groups:
        raw_placeholder = len(parameters) + 2
        raw_expression = f"plainto_tsquery('simple', ${raw_placeholder})"
        parameters.append(raw)

        if raw == expanded:
            expressions.append(raw_expression)
            continue

        expanded_placeholder = len(parameters) + 2
        expanded_expression = f"plainto_tsquery('simple', ${expanded_placeholder})"
        parameters.append(expanded)
        expressions.append(f"({raw_expression} || {expanded_expression})")

    return " && ".join(expressions), parameters


class MemoryRepository:
    """Владеет SQL и connection lifecycle для записей памяти."""

    def __init__(self, pool: asyncpg.Pool, settings: Settings) -> None:
        self._pool = pool
        self._settings = settings

    async def insert(self, record: MemoryInsertRecord) -> MemoryRecord:
        """Сохраняет полностью подготовленную ревизию памяти."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (
                    id,
                    logical_id,
                    revision,
                    supersedes_id,
                    scope,
                    project_id,
                    memory_type,
                    status,
                    content,
                    content_hash,
                    tags,
                    identifiers,
                    lexical_source,
                    lexical_profile_version,
                    embedding,
                    embedding_model,
                    embedding_profile_version,
                    provenance
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18::jsonb
                )
                RETURNING *;
                """,
                record.id,
                record.logical_id,
                record.revision,
                record.supersedes_id,
                record.scope.value,
                record.project_id,
                record.memory_type,
                record.status.value,
                record.content,
                record.content_hash,
                record.tags,
                record.identifiers,
                record.lexical_source,
                record.lexical_profile_version,
                record.embedding,
                record.embedding_model,
                record.embedding_profile_version,
                json.dumps(record.provenance),
            )

        if row is None:  # pragma: no cover - INSERT ... RETURNING всегда возвращает строку.
            raise RuntimeError("memory insert returned no row")
        return self._hydrate(row)

    async def get_by_id(self, id: UUID, project_id: str) -> MemoryRecord | None:
        """Возвращает видимую ревизию памяти по физическому UUID независимо от status."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM memories
                WHERE id = $1
                  AND (scope = 'global' OR project_id = $2);
                """,
                id,
                project_id,
            )
        return self._hydrate(row) if row is not None else None

    async def get_active_by_logical_id(
        self,
        logical_id: UUID,
        project_id: str,
    ) -> MemoryRecord | None:
        """Возвращает видимую активную ревизию логической памяти."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM memories
                WHERE logical_id = $1
                  AND status = 'active'
                  AND (scope = 'global' OR project_id = $2);
                """,
                logical_id,
                project_id,
            )
        return self._hydrate(row) if row is not None else None

    async def search_dense(
        self,
        query_embedding: list[float],
        project_id: str | None,
        limit: int,
        strategy: DenseSearchStrategy | None = None,
        *,
        memory_type: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Ищет активные memories по cosine distance в exact или HNSW режиме."""
        effective_strategy = strategy or self._settings.dense_retrieval_strategy
        if effective_strategy not in ("exact", "hnsw"):
            raise ValueError(f"unsupported dense search strategy: {effective_strategy}")

        # Dense и lexical branches получают независимые соединения при asyncio.gather().
        async with self._pool.acquire() as conn:
            # set_config(..., true) действует только до конца этой транзакции.
            async with conn.transaction():
                if effective_strategy == "exact":
                    await conn.execute("SELECT set_config('enable_indexscan', 'off', true)")
                else:
                    await conn.execute(
                        "SELECT set_config('hnsw.ef_search', $1, true)",
                        str(self._settings.hnsw_ef_search),
                    )
                    await conn.execute(
                        "SELECT set_config('hnsw.iterative_scan', $1, true)",
                        self._settings.hnsw_iterative_scan,
                    )

                rows = await conn.fetch(
                    """
                    SELECT *, embedding <=> $2 AS distance
                    FROM memories
                    WHERE status = 'active'
                      AND (scope = 'global' OR project_id = $1)
                      AND ($4::text IS NULL OR memory_type = $4)
                    ORDER BY distance ASC
                    LIMIT $3;
                    """,
                    project_id,
                    query_embedding,
                    limit,
                    memory_type,
                )

        results = [(self._hydrate(row), float(row["distance"])) for row in rows]
        # Не расширяем SQL ORDER BY: HNSW должен сортировать только по distance operator.
        # UUID стабилизирует ранги уже выбранного candidate set при равных distance.
        results.sort(key=lambda result: (result[1], result[0].id.int))
        return results

    async def search_lexical(
        self,
        lexical_query_groups: list[LexicalQueryGroup],
        project_id: str | None,
        limit: int,
        *,
        memory_type: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Ищет активные memories через syntax-safe PostgreSQL FTS query."""
        if not lexical_query_groups:
            return []

        query_expression, query_parameters = _build_lexical_query_expression(lexical_query_groups)
        limit_placeholder = len(query_parameters) + 2
        memory_type_placeholder = limit_placeholder + 1

        # Отдельное соединение позволяет запускать канал параллельно с dense retrieval.
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH query AS (
                    SELECT {query_expression} AS q
                )
                SELECT
                    m.*,
                    ts_rank_cd(m.lexical_text, query.q) AS lexical_score
                FROM memories AS m
                CROSS JOIN query
                WHERE m.lexical_text @@ query.q
                  AND m.status = 'active'
                  AND (m.scope = 'global' OR m.project_id = $1)
                  AND (${memory_type_placeholder}::text IS NULL
                       OR m.memory_type = ${memory_type_placeholder})
                ORDER BY lexical_score DESC, m.id ASC
                LIMIT ${limit_placeholder};
                """,
                project_id,
                *query_parameters,
                limit,
                memory_type,
            )

        return [(self._hydrate(row), float(row["lexical_score"])) for row in rows]

    @staticmethod
    def _hydrate(row: Mapping[str, Any]) -> MemoryRecord:
        """Проецирует DB row на domain model, исключая generated/search columns."""
        data = {field_name: row[field_name] for field_name in MemoryRecord.model_fields}

        # asyncpg декодирует jsonb в str по умолчанию, а codec pgvector — в Vector.
        if isinstance(data["provenance"], str):
            data["provenance"] = json.loads(data["provenance"])
        if isinstance(data["embedding"], Vector):
            data["embedding"] = data["embedding"].to_list()

        return MemoryRecord.model_validate(data)
