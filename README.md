# Orna Memory

Self-hosted context & memory backend для AI-агентов через Model Context Protocol (MCP).

Название отсылает к **Орне** — мечу Тетры из ирландского сказания *Cath Maige
Tuired*. Система сохраняет полезный опыт агента и возвращает его по запросу, не
превращая всю историю работы в бесконечный контекст.

## Как это работает

```text
AI-клиент → MCP → Orna Memory → PostgreSQL + pgvector
```

Клиент подключается к локальному MCP endpoint с Bearer token и передаёт
`X-Memory-Project`. Заголовок определяет проект: агент видит общие записи и память
своего проекта, но не записи других проектов.

Orna предоставляет три MCP-инструмента:

| Инструмент | Что делает |
|---|---|
| `memory_search` | Ищет подходящий прошлый опыт перед задачей |
| `memory_get` | Возвращает точную ревизию memory по UUID |
| `memory_add` | Сохраняет подтверждённый и повторно применимый опыт |

## Архитектура и стек

- **СУБД и поиск:** PostgreSQL 16 + `pgvector`
  - векторный поиск: HNSW-индекс с cosine distance;
  - лексический поиск: PostgreSQL FTS с конфигурацией `simple` без стемминга;
  - гибридное ранжирование: Reciprocal Rank Fusion (`k=60`);
  - два retrieval-канала выбирают по 20 кандидатов, итоговый ответ содержит до 5
    записей.
- **Эмбеддинги:** FastEmbed и `intfloat/multilingual-e5-large`
  - размерность вектора: 1024;
  - разные префиксы для запросов (`query:`) и памяти (`passage:`);
  - ONNX inference с двумя потоками и ограниченной конкуренцией;
  - зафиксированная ревизия модели хранится в persistent cache.
- **Модель памяти:** immutable revisions
  - отдельные `id` и `logical_id`, новые идентификаторы создаются как UUIDv7;
  - lifecycle status и связь `supersedes_id` хранят историю изменений;
  - содержимое ревизии защищено от изменения на уровне PostgreSQL;
  - `created_at` устанавливается PostgreSQL и остаётся authoritative timestamp;
  - content hash фиксирует canonical content, а tags и identifiers дополняют
    lexical index.
- **MCP-сервис:** Python 3.12+ и официальный MCP Python SDK v2
  - stateless Streamable HTTP;
  - Bearer authentication;
  - request-scoped project context через `X-Memory-Project`;
  - строгие схемы аргументов и ответы без embeddings и внутренних ranking scores.
- **Изоляция runtime:**
  - PostgreSQL и MCP публикуются только на `127.0.0.1`;
  - основной MCP-контейнер не имеет внешнего сетевого доступа;
  - model cache подключается к runtime в режиме read-only;
  - отдельный proxy публикует MCP endpoint, не получая секреты сервиса.

### Как выполняется поиск

`memory_search` одновременно использует два канала:

- семантический поиск по E5 embeddings через `pgvector`;
- лексический поиск PostgreSQL FTS для точного совпадения терминов и identifiers.

Для dense-канала запрос преобразуется в E5 embedding. Для lexical-канала строка
нормализуется в plain tokens, включая разбиение `camelCase`, `PascalCase` и
`snake_case` identifiers. Оба SQL-запроса выполняются параллельно через разные
соединения из connection pool.

Repository отбирает только active memories текущего проекта и записи с global
scope. Опциональный `memory_type` применяется до ограничения количества
кандидатов. Результаты объединяются через deterministic RRF и обрезаются до top-5.
Внутренние distance, rank и RRF score через MCP не возвращаются.

### Как выполняется запись

`memory_add` принимает `content`, `memory_type`, `tags` и `identifiers`. Scope,
project ID и базовый provenance принадлежат серверу и не управляются клиентом.

Перед inference и обращением к базе сервис:

1. Проверяет все сохраняемые строки на распространённые формы секретов.
2. Токенизирует `passage: <content>` без truncation и отклоняет ввод длиннее 512
   E5 tokens.
3. Вычисляет canonical content hash и собирает lexical source из текста, tags и
   identifiers.
4. Создаёт embedding и сохраняет первую active revision в PostgreSQL.

`memory_get` читает точную физическую ревизию по UUID. В отличие от поиска, такая
ревизия может иметь любой lifecycle status, но остаётся ограничена текущим
project context.

### Локальный runtime

При запуске Compose:

1. PostgreSQL запускается и проходит healthcheck.
2. Параллельно `model-cache-init` загружает и проверяет зафиксированную ревизию E5.
3. После готовности PostgreSQL `database-migrate` применяет SQL migrations.
4. После миграций и подготовки модели запускаются MCP-сервис и loopback proxy на
   `127.0.0.1:8000`.

Основной MCP-сервис работает без внешнего сетевого доступа и читает model cache в
режиме read-only. PostgreSQL и MCP публикуются только на `127.0.0.1`.

## Быстрый старт

### 1. Подготовьте окружение

```bash
cp .env.example .env
openssl rand -hex 32
```

Укажите в `.env` надёжный `POSTGRES_PASSWORD`, а результат `openssl` запишите в
`ORNA_MEMORY_TOKEN`.

### 2. Запустите сервисы

```bash
docker compose up -d
docker compose ps --all
```

При первом запуске потребуется интернет для загрузки модели. После подготовки
кэша основной runtime работает offline.

MCP endpoint: `http://127.0.0.1:8000/mcp`.

Для диагностики контейнеров используйте:

```bash
docker compose logs model-cache-init database-migrate orna-memory-mcp mcp-loopback
```

## Разработка и тесты

Зависимости управляются через [uv](https://docs.astral.sh/uv/):

```bash
cd orna-memory-mcp
uv lock --check
uv sync --frozen --all-groups
uv run python -m app.model_cache
uv run ruff format --check .
uv run ruff check .
uv run pytest tests/unit -v
uv run pytest tests/integration -v
```

Integration tests требуют PostgreSQL с применёнными migrations и подготовленный
E5 model cache.

## Лицензия

Проект распространяется под лицензией [GNU General Public License v3.0](LICENSE).
