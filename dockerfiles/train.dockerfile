FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.11-bookworm AS base

WORKDIR /workspace

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

ENV VIRTUAL_ENV=/workspace/.venv

COPY src src/
COPY configs configs/
COPY cloud cloud/
COPY .dvc .dvc/
COPY data data/
COPY entrypoint.sh entrypoint.sh
COPY README.md README.md
COPY LICENSE LICENSE

RUN mkdir -p models

RUN uv sync --frozen
RUN uv pip install --no-cache-dir --reinstall torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu118

RUN uv pip install --no-cache-dir "dvc[gs]"

ENV PATH="/usr/local/nvidia/bin:/workspace/.venv/bin:$PATH"

# Vertex injects the GPU driver into /usr/local/nvidia/lib64 at runtime, but the
# uv/bookworm base never puts it on the library path — so torch can't find
# libcuda.so.1 and torch.cuda.is_available() is False even with a CUDA build.
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib"

RUN dvc config core.no_scm true
