## Development

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Sync dependencies
uv sync

# Run linter and formatter
uv run ruff check .
uv run ruff format --check .

# Run test suite
uv run pytest
```
