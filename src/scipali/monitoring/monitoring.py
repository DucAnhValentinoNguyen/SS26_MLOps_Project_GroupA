"""Data-drift monitoring for ScienceQA inputs (M27).

Builds an Evidently data-drift report comparing a reference split (train) with a
current split (test, or live-collected inputs). We don't have raw tabular
features, so we derive lightweight ones from each sample — question length,
number of choices, hint/lecture presence, image dimensions, subject — which is
enough to catch distribution shift in the inputs the model sees.

Run (single-command Typer app, so no subcommand name):
    uv run --group serving python -m scipali.monitoring.monitoring
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import typer
from datasets import load_from_disk
from rich.logging import RichHandler

from scipali.data.data import DATASET_SUBSET, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger(__name__)

app = typer.Typer(help="Data-drift monitoring for ScienceQA inputs.")

RESULTS_DIR = Path("reports/monitoring")

# GCS feature tables /monitor/drift reads (mirrors the defaults in api.py).
_BUCKET = "gs://mlops-paligemma-west4/monitoring"
# Training-input distribution the drift check compares against.
REFERENCE_GCS = f"{_BUCKET}/reference.csv"
# Held-out demo distribution used only until production data is collected.
CURRENT_SAMPLE_GCS = f"{_BUCKET}/current_sample.csv"
# Where `collect` writes the production feature table that /monitor/drift
# compares against (the downstream consumer of the /predict logs, M27).
PRODUCTION_GCS = f"{_BUCKET}/current_production.csv"


def _features(split) -> pd.DataFrame:
    """Derive a tabular feature frame from a ScienceQA split for drift checks.

    Args:
        split: A HuggingFace Dataset split with image/question/choices/... cols.

    Returns:
        A DataFrame with one row per sample and numeric/categorical features.
    """
    rows = []
    for s in split:
        img = s.get("image")
        rows.append(
            {
                "question_char_len": len(s["question"]),
                "question_word_len": len(s["question"].split()),
                "num_choices": len(s["choices"]),
                "hint_present": int(bool(s.get("hint"))),
                "lecture_present": int(bool(s.get("lecture"))),
                "image_width": img.width if img is not None else 0,
                "image_height": img.height if img is not None else 0,
                "subject": s.get("subject", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def _features_from_log(payload: dict) -> dict | None:
    """Map one /predict structured-log event to the reference feature schema.

    Returns None for anything that is not a prediction event. ``subject`` is
    deliberately omitted: it is unknown at inference time (the model predicts
    the answer letter, not the subject), so emitting a constant placeholder
    would itself look like drift. /monitor/drift compares on the columns the
    production table actually carries.

    Args:
        payload: The decoded JSON object logged by /predict.

    Returns:
        A feature dict matching ``_features`` (minus ``subject``), or None.
    """
    if payload.get("event") != "prediction":
        return None
    question = payload.get("question", "") or ""
    px = payload.get("image_px") or [0, 0]
    width = int(px[0]) if len(px) > 0 else 0
    height = int(px[1]) if len(px) > 1 else 0
    return {
        "question_char_len": len(question),
        "question_word_len": len(question.split()),
        "num_choices": int(payload.get("n_choices", 0)),
        "hint_present": int(bool(payload.get("hint"))),
        "lecture_present": int(bool(payload.get("lecture"))),
        "image_width": width,
        "image_height": height,
    }


def _parse_log_entry(payload) -> dict | None:
    """Extract the JSON event object from one Cloud Logging entry payload.

    /predict logs ``log.info("prediction %s", json.dumps({...}))``. Depending
    on the logging handler this arrives either as a structured ``jsonPayload``
    dict or as a text payload like ``prediction {json}`` (possibly with
    soft-wrap newlines from the Rich handler — harmless, JSON treats them as
    whitespace). Parse both shapes defensively.

    Args:
        payload: ``entry.payload`` from a google-cloud-logging entry.

    Returns:
        The decoded event dict, or None if the line is not a JSON event.
    """
    if isinstance(payload, dict):
        return payload if "event" in payload else None
    if not isinstance(payload, str):
        return None
    start = payload.find("{")
    if start == -1:
        return None
    try:
        return json.loads(payload[start:])
    except json.JSONDecodeError:
        return None


def _write_gcs_csv(df: pd.DataFrame, uri: str) -> None:
    """Write a DataFrame as a CSV to a single gs:// blob."""
    from google.cloud import storage  # type: ignore[attr-defined]

    parsed = urlparse(uri)
    blob = storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/"))
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


@app.command()
def collect(
    project: str = typer.Option(..., envvar="GCP_PROJECT", help="GCP project id."),
    service: str = typer.Option("paligemma-api", help="Cloud Run service name."),
    output_gcs: str = typer.Option(PRODUCTION_GCS, help="gs:// CSV to write."),
    hours: int = typer.Option(168, help="Look back this many hours of logs."),
    limit: int = typer.Option(5000, help="Max log entries to scan."),
) -> None:
    """Materialise the 'current' production feature table from /predict logs.

    The missing downstream consumer (M27): /predict emits one structured
    'prediction' line per request to Cloud Logging; ``collect`` reads those
    back, derives the SAME features as the training reference, and writes them
    to a GCS CSV that /monitor/drift compares against. Without this step the
    drift endpoint can only self-compare held-out splits and never sees real
    production inputs.
    """
    from google.cloud import logging as cloud_logging  # type: ignore[attr-defined]

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    # No severity clause: /predict emits the event as a single-line JSON to
    # stdout (see api._log_prediction_event), which Cloud Run may surface as a
    # jsonPayload with INFO lifted out OR as a plain-text DEFAULT-severity line
    # depending on parsing — filtering on severity would drop the latter. The
    # ``"prediction"`` global match keeps the scan small (both shapes contain
    # it); _parse_log_entry still rejects anything that is not a real event.
    log_filter = (
        f'resource.type="cloud_run_revision" '
        f'resource.labels.service_name="{service}" '
        f'"prediction" '
        f'timestamp>="{since.isoformat()}"'
    )
    client = cloud_logging.Client(project=project)
    rows: list[dict] = []
    scanned = 0
    for entry in client.list_entries(
        filter_=log_filter, order_by=cloud_logging.DESCENDING, max_results=limit
    ):
        scanned += 1
        event = _parse_log_entry(entry.payload)
        if event is None:
            continue
        feats = _features_from_log(event)
        if feats is not None:
            rows.append(feats)

    if not rows:
        log.warning(
            "No /predict prediction events found in the last %dh (scanned %d "
            "entries). Send some traffic to /predict first.",
            hours,
            scanned,
        )
        raise typer.Exit(code=1)

    df = pd.DataFrame(rows)
    _write_gcs_csv(df, output_gcs)
    log.info(
        "Collected %d production samples (of %d log entries) -> %s",
        len(df),
        scanned,
        output_gcs,
    )


@app.command("seed-reference")
def seed_reference(
    processed_dir: Path = typer.Option(PROCESSED_DATA_DIR),
    reference: str = typer.Option("train", help="Split used as the drift reference."),
    sample: str = typer.Option(
        "test", help="Held-out split for the demo 'current' fallback table."
    ),
    reference_gcs: str = typer.Option(REFERENCE_GCS, help="gs:// reference CSV."),
    sample_gcs: str = typer.Option(
        CURRENT_SAMPLE_GCS, help="gs:// CSV for the demo sample."
    ),
) -> None:
    """(Re)generate the reference + demo-sample drift tables in GCS.

    Closes the reproducibility gap: the tables /monitor/drift reads were
    previously uploaded ad hoc, with no committed way to rebuild them. This
    derives both from the processed splits using the SAME ``_features`` as the
    rest of the pipeline and writes them to GCS, so the reference distribution
    is regenerable from committed code after any data change (e.g. a re-DVC).
    """
    dataset = load_from_disk(processed_dir / DATASET_SUBSET)
    ref_df = _features(dataset[reference])
    sample_df = _features(dataset[sample])
    _write_gcs_csv(ref_df, reference_gcs)
    _write_gcs_csv(sample_df, sample_gcs)
    log.info(
        "Seeded reference (%s, %d rows) -> %s", reference, len(ref_df), reference_gcs
    )
    log.info("Seeded sample (%s, %d rows) -> %s", sample, len(sample_df), sample_gcs)


@app.command()
def drift(
    processed_dir: Path = typer.Option(PROCESSED_DATA_DIR),
    reference: str = typer.Option("train", help="Reference split."),
    current: str = typer.Option("test", help="Current split to compare."),
    output_dir: Path = typer.Option(RESULTS_DIR, "--output-dir", "-o"),
) -> None:
    """Generate an Evidently data-drift report (reference vs current split)."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    dataset = load_from_disk(processed_dir / DATASET_SUBSET)
    ref_df = _features(dataset[reference])
    cur_df = _features(dataset[current])
    log.info(
        "Reference (%s): %d rows | Current (%s): %d rows",
        reference,
        len(ref_df),
        current,
        len(cur_df),
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "drift_report.html"
    report.save_html(str(html_path))

    result = report.as_dict()["metrics"][0]["result"]
    log.info(
        "Dataset drift detected: %s | drifted columns: %d/%d",
        result.get("dataset_drift"),
        result.get("number_of_drifted_columns"),
        result.get("number_of_columns"),
    )
    log.info("Saved drift report to %s", html_path)


if __name__ == "__main__":
    app()
