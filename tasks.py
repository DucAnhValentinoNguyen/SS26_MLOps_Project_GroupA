"""Task definitions for invoke."""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "scipali"
PYTHON_VERSION = "3.11.0"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Download and preprocess data."""
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.data.data download",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.data.data preprocess",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context, config: str = "train") -> None:
    """Train model."""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/models/train.py --config-name {config}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f"
        f" dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f "
        f"dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t predict:latest . -f "
        f"dockerfiles/predict.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,
    )


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run(
        "uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS
    )
