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

`memory_search` принимает natural-language `query` и опциональный exact-фильтр
`memory_type`. MCP фиксирует итоговый `limit=5`; клиент не управляет внутренними
retrieval limits и ranking parameters.

Поиск проходит в четыре этапа:

1. **Подготовка запроса.** Для dense-канала сервис вычисляет 1024-мерный E5
   embedding с префиксом `query:`. Для lexical-канала запрос NFC-нормализуется и
   разбивается на независимые semantic groups.
2. **Параллельный retrieval.** Dense и lexical SQL-запросы одновременно получают
   отдельные соединения из connection pool и выбирают до 20 кандидатов каждый.
3. **Объединение.** Два упорядоченных списка сливаются через Reciprocal Rank Fusion
   с `k=60`.
4. **Публичная проекция.** Первые пять memories возвращаются через MCP без
   embeddings, distance, channel ranks и RRF score.

Dense-канал сравнивает embedding запроса с сохранёнными vectors по cosine distance
через `pgvector`. Repository поддерживает exact и HNSW retrieval; выбранная
стратегия не меняет публичный MCP contract.

Lexical-канал использует PostgreSQL FTS с конфигурацией `simple`. Индексируемый
`lexical_text` строится из `content`, tags и identifiers. Technical identifiers
сохраняются и в исходной, и в раскрытой форме: `ResponseProviderExecutor` также
индексируется как `response provider executor`, `routing_pool` — как
`routing pool`, а `Application.php` — как `application php`.

Для каждой semantic group raw и expanded формы являются альтернативами через OR,
а независимые группы обязательны одновременно через AND. Концептуально запрос
`ResponseProviderExecutor timeout` превращается в:

```text
(responseproviderexecutor OR response provider executor) AND timeout
```

Фактическое SQL expression собирается из фиксированных
`plainto_tsquery('simple', ...)`, а все пользовательские значения передаются как
bound parameters. Поэтому punctuation в запросе не интерпретируется как `tsquery`
syntax. Если после нормализации lexical groups не осталось, lexical-канал возвращает
пустой список, но dense-канал продолжает работать.

Оба канала до `LIMIT` отбирают только memories со `status = 'active'`, видимые
текущему проекту: записи его `project_id` и записи со `scope = 'global'`.
Опциональный `memory_type` также применяется до выбора кандидатов.

Для каждого кандидата итоговый score равен сумме вкладов каналов, в которых он
встретился:

```text
RRF score = Σ 1 / (60 + rank_in_channel)
```

Например, memory с dense rank 3 и lexical rank 1 получает
`1 / 63 + 1 / 61`. Совпадение в обоих каналах обычно поднимает результат выше
кандидата, найденного только одним каналом. При равном score порядок стабилизируют
лучший channel rank, затем `logical_id` и физический `id`.

Search выполняет retrieval и ранжирование, но не устанавливает истинность или
актуальность найденного утверждения. Клиент должен сверять результат с текущим
code, schema, tests и authoritative documentation.

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
