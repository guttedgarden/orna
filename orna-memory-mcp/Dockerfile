# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ARG UV_VERSION=0.12.2

ENV PATH="/opt/orna/orna-memory-mcp/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir "uv==${UV_VERSION}" \
    && groupadd --system --gid 10001 orna \
    && useradd --system --uid 10001 --gid orna --create-home orna \
    && mkdir -p /opt/orna/orna-memory-mcp /var/cache/orna/fastembed \
    && chown -R orna:orna /opt/orna /var/cache/orna

WORKDIR /opt/orna/orna-memory-mcp

COPY README.md /opt/orna/README.md
COPY orna-memory-mcp/pyproject.toml orna-memory-mcp/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=orna:orna orna-memory-mcp/app ./app

USER orna

CMD ["python", "-m", "app.model_cache"]
