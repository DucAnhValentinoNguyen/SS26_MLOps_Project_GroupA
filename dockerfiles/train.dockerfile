FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.11-bookworm AS base

WORKDIR /workspace

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

ENV VIRTUAL_ENV=/workspace/.venv

COPY src src/
COPY configs configs/
COPY .dvc .dvc/
COPY data data/
COPY entrypoint.sh entrypoint.sh
COPY README.md README.md
COPY LICENSE LICENSE

RUN mkdir -p models

RUN uv sync --frozen
RUN uv pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-cache-dir

RUN uv pip install --no-cache-dir "dvc[gs]"

ENV PATH="/workspace/.venv/bin:$PATH"

RUN dvc config core.no_scm true
