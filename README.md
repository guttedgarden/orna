# Orna Memory

Self-hosted context & memory backend для AI-агентов через Model Context Protocol (MCP).

Название отсылает к **Орне** — мечу Тетры из ирландского сказания *Cath Maige Tuired*. Будучи обнажённым и очищенным, меч рассказывал о совершённых им деяниях. Система возвращает накопленный опыт и контекст по запросу, а не хранит весь шум прошлого.

---

## Архитектура и стек

* **СУБД и поиск:** PostgreSQL 16 + `pgvector`
  * Векторный поиск: HNSW-индекс (cosine distance)
  * Лексический поиск: PostgreSQL FTS (конфиг `simple` без стемминга для точного матчинга идентификаторов)
  * Гибридное ранжирование: Reciprocal Rank Fusion (RRF, $k=60$)
* **Эмбеддинги:** FastEmbed (`intfloat/multilingual-e5-large`, 2 потока ONNX)
* **Сервис:** Python 3.12+, официальный MCP Python SDK v2 (Streamable HTTP)
* **Изоляция:** порты публикуются только на `127.0.0.1` (loopback-only)

---

## Быстрый старт

### 1. Подготовка окружения

Скопируйте шаблон секретов и укажите пароль для базы данных:

```bash
cp .env.example .env
# Отредактируйте .env и задайте надежный POSTGRES_PASSWORD
```

### 2. Запуск базы данных

```bash
docker compose up -d
docker compose ps
```

PostgreSQL с `pgvector` будет доступен локально по адресу `127.0.0.1:5432`.

### 3. Разработка и тесты

Для управления зависимостями используется [uv](https://docs.astral.sh/uv/):

```bash
cd orna-memory-mcp

# Установка зависимостей и виртуального окружения
uv sync

# Проверка линтером и форматтером
uv run ruff check .
uv run ruff format --check .

# Запуск тестов
uv run pytest
```

---

## Лицензия

Проект распространяется под лицензией [GNU General Public License v3.0](LICENSE).
