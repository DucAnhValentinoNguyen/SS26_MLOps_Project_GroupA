"""FastAPI inference service for PaliGemma2 ScienceQA predictions."""

import base64
import io
import json
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from scipali.models.model import PaliGemmaModule
from scipali.serving.predict import load_model, predict_single

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)


def _log_prediction_event(event: dict) -> None:
    """Emit one structured prediction event as a single-line JSON to stdout.

    Written directly to stdout, NOT through the Rich-formatted app logger: Rich
    renders at a fixed 80-column width and Cloud Run captures each wrapped
    visual line as a SEPARATE log entry, so a long event line is truncated and
    fragmented across entries (and gets a right-aligned ``api.py:NNN`` suffix) —
    which makes it impossible for ``monitoring.collect`` to reassemble. A lone
    single-line JSON object is instead parsed by Cloud Run into
    ``LogEntry.jsonPayload`` (with ``severity`` lifted out), and read back
    intact by collect. The ``message`` key gives the Logs Explorer a readable
    summary without affecting the parsed payload.
    """
    record = {"severity": "INFO", "message": "prediction", **event}
    print(json.dumps(record), flush=True)


_module: PaliGemmaModule | None = None
_load_lock = threading.Lock()

# Drift detection (M27): reference = training-input feature distribution,
# current_sample = a held-out distribution to compare against. Both live in GCS
# so the container needs neither the dataset nor a rebuild to refresh them.
_BUCKET = "gs://mlops-paligemma-west4/monitoring"
REFERENCE_GCS = os.environ.get("REFERENCE_GCS", f"{_BUCKET}/reference.csv")
# Real collected production inputs (materialised from /predict logs by
# `python -m scipali.monitoring.monitoring collect`). This is the default "current"
# distribution so the endpoint is not a self-comparison once traffic exists.
PRODUCTION_GCS = os.environ.get("PRODUCTION_GCS", f"{_BUCKET}/current_production.csv")
# Held-out demo distribution used only as a fallback before any production
# data has been collected (keeps the endpoint working out of the box).
CURRENT_SAMPLE_GCS = os.environ.get(
    "CURRENT_SAMPLE_GCS", f"{_BUCKET}/current_sample.csv"
)


def _fetch_gcs_dir(uri: str) -> Path:
    """Download a GCS directory (e.g. the production adapter) to a temp dir.

    Lets the serving container start from the stable GCS path
    (gs://mlops-paligemma-west4/models/production) instead of baking the
    adapter into the image — promotion then needs no rebuild/redeploy.

    Args:
        uri: gs://bucket/prefix directory holding the adapter files.

    Returns:
        Local directory containing the downloaded files.

    Raises:
        FileNotFoundError: If no objects exist under the prefix.
    """
    from google.cloud import storage  # type: ignore[attr-defined]

    parsed = urlparse(uri)
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    dest_root = Path(tempfile.mkdtemp(prefix="adapter-"))
    client = storage.Client()
    count = 0
    for blob in client.list_blobs(parsed.netloc, prefix=prefix):
        rel = blob.name[len(prefix) :]
        if not rel:
            continue
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        count += 1
    if not count:
        raise FileNotFoundError(f"No objects found under {uri}")
    log.info("Fetched %d adapter files from %s", count, uri)
    return dest_root


def _read_gcs_csv(uri: str):
    """Read a single gs:// CSV blob into a pandas DataFrame."""
    import io as _io

    import pandas as pd
    from google.cloud import storage  # type: ignore[attr-defined]

    parsed = urlparse(uri)
    blob = storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/"))
    return pd.read_csv(_io.StringIO(blob.download_as_text()))


def _resolve_current(current_gcs: str | None):
    """Pick the 'current' distribution for the drift check and report its URI.

    Priority: an explicit ``current_gcs`` override, else the collected
    production table (PRODUCTION_GCS) when it exists and is non-empty, else the
    held-out demo sample (CURRENT_SAMPLE_GCS). The fallback chain means the
    endpoint defaults to real production inputs once any have been collected,
    instead of silently self-comparing two slices of the same dataset.

    Args:
        current_gcs: Optional explicit gs:// CSV override.

    Returns:
        Tuple of (DataFrame, gs:// URI actually used).
    """
    if current_gcs:
        return _read_gcs_csv(current_gcs), current_gcs
    try:
        df = _read_gcs_csv(PRODUCTION_GCS)
        if len(df) > 0:
            return df, PRODUCTION_GCS
        log.warning(
            "Production drift table %s is empty — falling back to %s",
            PRODUCTION_GCS,
            CURRENT_SAMPLE_GCS,
        )
    except Exception as exc:  # noqa: BLE001 - any read failure -> fall back
        log.warning(
            "No production drift table at %s (%s) — falling back to %s",
            PRODUCTION_GCS,
            exc,
            CURRENT_SAMPLE_GCS,
        )
    return _read_gcs_csv(CURRENT_SAMPLE_GCS), CURRENT_SAMPLE_GCS


def _load_module() -> PaliGemmaModule | None:
    """Resolve ``CHECKPOINT_PATH`` and load the model, or return None.

    The path may be a LoRA adapter directory, a full .ckpt file, or a gs://
    directory (downloaded to a temp dir first). Returns None (rather than
    raising) when the variable is unset or the path is missing, so the service
    can still start and report its state via the health endpoint.
    """
    checkpoint_env = os.environ.get("CHECKPOINT_PATH", "")
    if not checkpoint_env:
        log.warning("CHECKPOINT_PATH not set — /predict will return 503.")
        return None
    if checkpoint_env.startswith("gs://"):
        log.info("CHECKPOINT_PATH is a GCS uri — downloading %s", checkpoint_env)
        checkpoint_path = _fetch_gcs_dir(checkpoint_env)
    else:
        checkpoint_path = Path(checkpoint_env)
    if not checkpoint_path.exists():
        log.warning("CHECKPOINT_PATH is set but path not found: %s", checkpoint_path)
        return None
    log.info("Loading model from %s ...", checkpoint_path)
    module = load_model(checkpoint_path)
    log.info("Model ready.")
    return module


def _ensure_loaded() -> PaliGemmaModule | None:
    """Return the model, loading it once on first use (thread-safe).

    Used by lazy loading: on Cloud Run, downloading + loading a 3B model can
    exceed the startup-probe window, so we defer it to the first request
    instead of blocking container startup.
    """
    global _module
    if _module is None:
        with _load_lock:
            if _module is None:
                _module = _load_module()
    return _module


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, unless ``LAZY_LOAD`` defers it to first use.

    Default (local, tests): eager load so the model is ready before serving.
    With ``LAZY_LOAD=1`` (Cloud Run): skip the load so the container binds the
    port immediately and passes the startup probe; the first /predict request
    triggers the (slow) load via ``_ensure_loaded``.

    Args:
        app: The FastAPI application instance (required by the lifespan protocol).
    """
    global _module
    if os.environ.get("LAZY_LOAD", "").lower() in ("1", "true", "yes"):
        log.info("LAZY_LOAD set — deferring model load to the first request.")
    elif os.environ.get("CHECKPOINT_PATH", ""):
        _module = _load_module()
    else:
        log.warning("CHECKPOINT_PATH not set — /predict will return 503.")

    yield

    _module = None
    log.info("Model released.")


app = FastAPI(
    title="PaliGemma2 ScienceQA API",
    description="Single-sample inference endpoint for "
    "PaliGemma2 fine-tuned on ScienceQA.",
    version="0.1.0",
    lifespan=lifespan,
)

# System metrics (M28): expose Prometheus metrics at /metrics — request counts,
# latency histograms, in-progress requests, request/response sizes. Cloud Run /
# Managed Prometheus can scrape it. Wrapped defensively so the API still imports
# without the optional dependency (only the deployed image and the `monitoring`
# group install prometheus-fastapi-instrumentator).
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )
except ImportError:  # pragma: no cover - metrics are an optional extra
    log.warning("prometheus-fastapi-instrumentator not installed — /metrics disabled.")


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint.

    Attributes:
        question: The question text.
        choices: List of answer choice strings (2–10 items).
        hint: Optional hint text appended to the prompt.
        lecture: Optional lecture text appended to the prompt.
        image_b64: Required base64-encoded image string (JPEG or PNG).
        max_new_tokens: Maximum number of tokens the model may generate.
    """

    question: str = Field(..., min_length=1, description="Question text.")
    choices: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Answer choices, e.g. ['True', 'False'] or ['A', 'B', 'C'].",
    )
    hint: str = Field(default="", description="Optional hint text. Skipped if empty.")
    lecture: str = Field(
        default="", description="Optional lecture text. Skipped if empty."
    )
    image_b64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded image (JPEG or PNG).",
    )
    max_new_tokens: int = Field(
        default=10, ge=1, le=128, description="Max tokens to generate."
    )


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint.

    Attributes:
        prediction: Predicted answer letter, e.g. "A" or "B".
    """

    prediction: str = Field(..., description="Predicted answer letter (A/B/C/D/...).")


class DriftResponse(BaseModel):
    """Response body for the /monitor/drift endpoint.

    Attributes:
        dataset_drift: Whether Evidently flags overall dataset drift.
        n_drifted_columns: Number of input features detected as drifted.
        n_columns: Total number of features compared.
        reference_rows: Row count of the reference distribution.
        current_rows: Row count of the current distribution.
        current_source: gs:// URI of the current distribution actually used.
    """

    dataset_drift: bool = Field(..., description="Overall drift verdict.")
    n_drifted_columns: int = Field(..., description="Features detected as drifted.")
    n_columns: int = Field(..., description="Features compared.")
    reference_rows: int = Field(..., description="Reference distribution size.")
    current_rows: int = Field(..., description="Current distribution size.")
    current_source: str = Field(..., description="gs:// URI of the current table used.")


@app.get("/", summary="Health check")
def root() -> dict[str, str]:
    """Return service status and whether the model is loaded.

    Returns:
        A dict with ``status`` and ``model_loaded`` keys.
    """
    return {
        "status": "ok",
        "model_loaded": str(_module is not None),
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run inference on a single sample",
)
def predict(request: PredictRequest) -> PredictResponse:
    """Run single-sample inference and return the predicted answer letter.

    The image must be base64-encoded JPEG or PNG.
    The prediction is a single letter (A, B, C, ...) corresponding to the
    index of the predicted choice in ``request.choices``.

    Args:
        request: PredictRequest containing the question, choices, and
            optional image and hint.

    Returns:
        PredictResponse with the predicted answer letter.

    Raises:
        HTTPException 503: Model checkpoint has not been loaded.
        HTTPException 400: Image decoding failed.
    """
    module = _ensure_loaded()
    if module is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not loaded. "
            "Please set CHECKPOINT_PATH and restart the service.",
        )

    # Decode the image (required — PaliGemma is image-conditioned)
    try:
        image_bytes = base64.b64decode(request.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image_b64: {exc}",
        ) from exc

    # Only forward optional fields that were actually provided,
    # matching whatever columns survived --drop-cols during preprocessing.
    prompt_kwargs: dict[str, str] = {}
    if request.hint:
        prompt_kwargs["hint"] = request.hint
    if request.lecture:
        prompt_kwargs["lecture"] = request.lecture

    prediction = predict_single(
        module=module,
        image=image,
        question=request.question,
        choices=request.choices,
        max_new_tokens=request.max_new_tokens,
        **prompt_kwargs,
    )

    # Input-output collection (M27): one structured JSON line per prediction ->
    # Cloud Logging, queryable later for data-drift monitoring. The image bytes
    # are not logged (size); its dimensions are.
    _log_prediction_event(
        {
            "event": "prediction",
            "question": request.question,
            "n_choices": len(request.choices),
            "hint": bool(request.hint),
            "lecture": bool(request.lecture),
            "image_px": list(image.size),
            "prediction": prediction,
        }
    )
    return PredictResponse(prediction=prediction)


@app.get(
    "/monitor/drift",
    response_model=DriftResponse,
    summary="Data-drift check (Evidently) on the input distribution",
)
def monitor_drift(
    current_gcs: str | None = None,
) -> DriftResponse:
    """Run an Evidently data-drift check: reference (training inputs) vs current.

    Both feature tables are read from GCS, so the model container needs neither
    the dataset nor a rebuild. With no ``current_gcs`` the endpoint compares
    against the collected production inputs (PRODUCTION_GCS, materialised from
    the /predict logs by ``python -m scipali.monitoring.monitoring collect``), falling
    back to a held-out demo sample only until production data exists. Pass
    ``current_gcs`` to compare against an explicit table.

    The comparison is restricted to the columns both tables share, so a
    production table that lacks a train-only column (e.g. ``subject``, unknown
    at inference time) does not error or register as spurious drift.

    Args:
        current_gcs: Optional gs:// CSV of the current input-feature distribution.

    Returns:
        DriftResponse with the overall verdict, drifted-column counts, and the
        gs:// URI of the current distribution that was actually used.

    Raises:
        HTTPException 500: If the drift computation fails (e.g. GCS/Evidently).
    """
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        ref = _read_gcs_csv(REFERENCE_GCS)
        cur, source = _resolve_current(current_gcs)
        # Compare only on shared columns (production inputs can't carry
        # train-only columns like subject).
        common = [c for c in ref.columns if c in cur.columns]
        ref, cur = ref[common], cur[common]
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        result = report.as_dict()["metrics"][0]["result"]
    except Exception as exc:  # noqa: BLE001 - surface any failure as 500
        raise HTTPException(
            status_code=500, detail=f"drift check failed: {exc}"
        ) from exc

    return DriftResponse(
        dataset_drift=bool(result["dataset_drift"]),
        n_drifted_columns=int(result["number_of_drifted_columns"]),
        n_columns=int(result["number_of_columns"]),
        reference_rows=len(ref),
        current_rows=len(cur),
        current_source=source,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
