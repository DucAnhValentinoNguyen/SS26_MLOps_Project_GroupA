FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "project_name.api:app", "--host", "0.0.0.0", "--port", "8000"]
