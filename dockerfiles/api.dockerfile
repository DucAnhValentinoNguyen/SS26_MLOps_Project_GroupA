FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# --group monitoring adds evidently (/monitor/drift) + the Prometheus
# instrumentator (/metrics, M27/M28). --no-dev keeps test/docs/lint tooling out
# of the serving image (the API imports only base + monitoring deps at runtime).
RUN uv sync --frozen --no-install-project --no-dev --group monitoring

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen --no-dev --group monitoring

# Cloud Run injects $PORT (8080); default to 8000 locally. Shell form so the
# variable expands at runtime. --no-sync: deps are already frozen-synced above,
# so don't hit the network on start.
ENV PORT=8000
EXPOSE 8000

ENTRYPOINT ["sh", "-c", "uv run --no-sync uvicorn project_name.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
