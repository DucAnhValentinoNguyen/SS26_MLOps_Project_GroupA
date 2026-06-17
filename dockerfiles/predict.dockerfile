FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# --no-dev: inference needs only base deps (torch/transformers/peft), not the
# test/docs/lint tooling in the dev group.
RUN uv sync --frozen --no-install-project --no-dev

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen --no-dev

# --no-sync: deps are already frozen-synced above, so use that lean venv and
# don't re-sync (which would re-add the dev group and hit the network) at start.
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "scipali.serving.predict"]
