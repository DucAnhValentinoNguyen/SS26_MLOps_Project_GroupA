"""Tests for the evaluate CLI (mocked model + dataloader — no GPU, data, or net).

Mirrors the mocking style of test_predict.py: patch ``load_model`` and
``DataModule`` so the real evaluation loop, exact-match scoring (via the real
``extract_answer_letter``), by-subject aggregation, and JSON writing all run.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from typer.testing import CliRunner

from scipali.models.evaluate import app

runner = CliRunner()


def _fake_batch(targets: list[str], subjects: list[str]) -> dict:
    """One generation-style test batch (answer-free input_ids + ground truth)."""
    n = len(targets)
    return {
        "input_ids": torch.zeros((n, 5), dtype=torch.long),
        "attention_mask": None,
        "pixel_values": None,
        "answer_texts": targets,
        "subjects": subjects,
    }


def _fake_model(decoded_preds: list[str]) -> MagicMock:
    """A model whose generate→batch_decode yields ``decoded_preds``."""
    model = MagicMock()
    model.device = torch.device("cpu")  # real device so tensor.to(...) works
    model.model.generate.return_value = torch.zeros(
        (len(decoded_preds), 8), dtype=torch.long
    )
    model.processor.batch_decode.return_value = decoded_preds
    return model


def test_evaluate_writes_results_and_accuracy(tmp_path: Path) -> None:
    """Evaluate runs inference, computes exact-match accuracy, writes JSON."""
    targets = ["A", "C"]
    preds = ["A", "B"]  # A==A correct, B!=C wrong -> 1/2 = 0.5
    subjects = ["natural science", "social science"]
    model = _fake_model(preds)
    dm = MagicMock()
    dm.test_dataloader.return_value = [_fake_batch(targets, subjects)]

    out = tmp_path / "eval_results.json"
    with (
        patch("scipali.models.evaluate.load_model", return_value=model),
        patch("scipali.models.evaluate.DataModule", return_value=dm),
    ):
        result = runner.invoke(
            app, ["fake.ckpt", "--output-path", str(out), "--by-subject"]
        )

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["total_samples"] == 2
    assert data["total_correct"] == 1
    assert data["accuracy"] == 0.5
    assert data["split"] == "test"
    assert data["by_subject"]["natural science"]["accuracy"] == 1.0
    assert data["by_subject"]["social science"]["accuracy"] == 0.0


def test_evaluate_limit_batches_stops_early(tmp_path: Path) -> None:
    """--limit-batches halts the loop so only the first batch is scored."""
    model = _fake_model(["A", "A"])
    dm = MagicMock()
    dm.test_dataloader.return_value = [
        _fake_batch(["A", "A"], ["natural science", "natural science"]),
        _fake_batch(["B", "B"], ["social science", "social science"]),
    ]

    out = tmp_path / "eval.json"
    with (
        patch("scipali.models.evaluate.load_model", return_value=model),
        patch("scipali.models.evaluate.DataModule", return_value=dm),
    ):
        result = runner.invoke(
            app, ["fake.ckpt", "--output-path", str(out), "--limit-batches", "1"]
        )

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["total_samples"] == 2  # only the first batch was evaluated
    assert data["accuracy"] == 1.0
