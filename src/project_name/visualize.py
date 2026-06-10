"""Visualization utilties for inspecting ScienceQA model predictions."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import typer
from datasets import load_from_disk
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

app = typer.Typer(help="Visualize model predictions and evaluation results.")

PROCESSED_DATA_DIR = Path("data/processed/ScienceQA-IMG")
RESULTS_DIR = Path("reports/figures")


def plot_accuracy_by_subject(
    results_path: Path,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    """Plot per-subject accuracy as a horizontal bar chart.

    Reads the JSON report produced by evaluate.py and groups accuracy
    by the `subject` field present in each sample.

    Args:
        results_path: Path to the evaluate.py JSON results file.
        output_dir: Directory where the figure is saved.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open() as f:
        results = json.load(f)

    subject_counts: dict[str, list[int]] = {}
    for item in results["samples"]:
        subject = item.get("subject", "unknown")
        correct = int(item["correct"])
        if subject not in subject_counts:
            subject_counts[subject] = [0, 0]
        subject_counts[subject][0] += correct
        subject_counts[subject][1] += 1

    subjects = sorted(subject_counts.keys())
    accuracies = [subject_counts[s][0] / subject_counts[s][1] * 100 for s in subjects]

    fig, ax = plt.subplots(figsize=(8, max(3, len(subjects) * 0.5)))
    bars = ax.barh(subjects, accuracies, color="steelblue")
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Subject Accuracy")
    ax.set_xlim(0, 110)
    fig.tight_layout()

    out_path = output_dir / "accuracy_by_subject.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved accuracy by subject plot to %s", out_path)
    return out_path


def plot_error_samples(
    results_path: Path,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = RESULTS_DIR,
    n_samples: int = 6,
    split: str = "test",
) -> Path:
    """Plot a grid of incorrectly predicted samples with image and text.

    Each cell shows the image (if available), the question, the ground-truth
    answer, and the model's prediction, to aid qualitative error analysis.

    Args:
        results_path: Path to the evaluate.py JSON results file.
        processed_data_dir: Root directory of the processed dataset.
        output_dir: Directory where the figure is saved.
        n_samples: Maximum number of error samples to display.
        split: Dataset split to load images from (default: "test").

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open() as f:
        results = json.load(f)

    # Collect incorrect predictions
    errors = [s for s in results["samples"] if not s["correct"]]
    if not errors:
        log.warning("No errors found in results - all predictions were correct.")
        out_path = output_dir / "error_samples.png"
        return out_path

    errors = errors[:n_samples]

    dataset = load_from_disk(str(processed_data_dir))
    ds_split = dataset[split]

    n_cols = 3
    n_rows = (len(errors) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for ax, err in zip(axes, errors):
        # evaluate.py records the row index in the (unshuffled) split
        idx = err.get("index")
        row = ds_split[idx] if idx is not None and idx < len(ds_split) else None
        image = row["image"] if row is not None else None

        if image is not None:
            ax.imshow(image)
        else:
            ax.set_facecolor("#f0f0f0")
            ax.text(
                0.5,
                0.5,
                "No image",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )

        question = row["question"] if row is not None else ""
        label = (
            f"Q: {question[:60]}{'...' if len(question) > 60 else ''}\n"
            f"GT: {err.get('label', '?')} | Pred: {err.get('prediction', '?')}"
        )
        ax.set_xlabel(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for ax in axes[len(errors) :]:
        ax.set_visible(False)

    fig.suptitle("Incorrectly Predicted Samples", fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = output_dir / "error_samples.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved error samples plot to %s", out_path)
    return out_path


def plot_prediction_length_distribution(
    results_path: Path,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    """Plot a histogram of predicted answer token lengths.

    Useful for diagnosing whether the model is truncating answers or
    generating excessively long outputs.

    Args:
        results_path: Path to the evaluate.py JSON results file.
        output_dir: Directory where the figure is saved.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open() as f:
        results = json.load(f)

    lengths = [len(s.get("prediction", "").split()) for s in results["samples"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=20, color="steelblue", edgecolor="black")
    ax.set_xlabel("Predicted Answer Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Length Distribution")
    fig.tight_layout()

    out_path = output_dir / "prediction_length_dist.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved prediction length distribution plot to %s", out_path)
    return out_path


@app.command()
def subject_accuracy(
    results_path: Path = typer.Argument(
        ..., help="Path to the evaluate.py JSON results."
    ),
    output_dir: Path = typer.Option(RESULTS_DIR, "--output-dir", "-o"),
) -> None:
    """Plot per-subject accuracy from an evaluate.py results file."""
    out = plot_accuracy_by_subject(results_path, output_dir)
    typer.echo(f"Saved to {out}")


@app.command()
def error_samples(
    results_path: Path = typer.Argument(..., help="Path to evaluate.py JSON results."),
    data_dir: Path = typer.Option(PROCESSED_DATA_DIR, "--data-dir"),
    output_dir: Path = typer.Option(RESULTS_DIR, "--output-dir", "-o"),
    n: int = typer.Option(
        6, "--n-samples", "-n", help="Number of error samples to show."
    ),
    split: str = typer.Option("test", "--split", "-s"),
) -> None:
    """Plot a grid of incorrectly predicted samples for qualitative analysis."""
    out = plot_error_samples(results_path, data_dir, output_dir, n, split)
    typer.echo(f"Saved to {out}")


@app.command()
def pred_lengths(
    results_path: Path = typer.Argument(..., help="Path to evaluate.py JSON results."),
    output_dir: Path = typer.Option(RESULTS_DIR, "--output-dir", "-o"),
) -> None:
    """Plot the distribution of predicted answer lengths."""
    out = plot_prediction_length_distribution(results_path, output_dir)
    typer.echo(f"Saved to {out}")


if __name__ == "__main__":
    app()
