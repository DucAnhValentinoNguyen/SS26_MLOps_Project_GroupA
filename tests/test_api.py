"""Tests for the FastAPI inference API."""

import base64
import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from project_name.api import app


@pytest.fixture(autouse=True)
def _no_checkpoint_env(monkeypatch):
    """Ensure CHECKPOINT_PATH is unset so lifespan doesn't load a real model."""
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)


def _make_image_b64() -> str:
    """Return a base64-encoded tiny PNG for requests that need a valid image."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


VALID_IMAGE_B64 = _make_image_b64()


def test_root_return_200() -> None:
    """Root endpoint returns HTTP 200 and expected keys."""
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_predict_missing_required_fields_returns_422() -> None:
    """Predict endpoint returns 422 when required fields are absent."""
    with TestClient(app) as client:
        response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_too_few_choices_returns_422() -> None:
    """Predict endpoint returns 422 when fewer than 2 choices are provided."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"question": "Is water wet?", "choices": ["Yes"]},
        )
    assert response.status_code == 422


def test_predict_without_model_returns_503() -> None:
    """Predict endpoint returns 503 when no checkpoint is loaded."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "question": "Is water wet?",
                "choices": ["Yes", "No"],
                "image_b64": VALID_IMAGE_B64,
            },
        )
    assert response.status_code == 503


def test_lifespan_fetches_gcs_checkpoint(monkeypatch, tmp_path) -> None:
    """A gs:// CHECKPOINT_PATH is downloaded via _fetch_gcs_dir, then loaded."""
    monkeypatch.setenv("CHECKPOINT_PATH", "gs://bucket/models/production")
    with (
        patch("project_name.api._fetch_gcs_dir", return_value=tmp_path) as mock_fetch,
        patch("project_name.api.load_model") as mock_load,
    ):
        with TestClient(app) as client:
            response = client.get("/")
        mock_fetch.assert_called_once_with("gs://bucket/models/production")
        mock_load.assert_called_once_with(tmp_path)
    assert response.json()["model_loaded"] == "True"


def test_lifespan_local_path_skips_gcs_fetch(monkeypatch, tmp_path) -> None:
    """A local CHECKPOINT_PATH never touches GCS."""
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path))
    with (
        patch("project_name.api._fetch_gcs_dir") as mock_fetch,
        patch("project_name.api.load_model") as mock_load,
    ):
        with TestClient(app):
            pass
        mock_fetch.assert_not_called()
        mock_load.assert_called_once()


def test_predict_returns_prediction() -> None:
    """Predict endpoint returns a prediction letter when model is loaded."""
    with (
        patch("project_name.api._module", new=object()),
        patch("project_name.api.predict_single", return_value="A"),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "question": "Is water wet?",
                    "choices": ["Yes", "No"],
                    "image_b64": VALID_IMAGE_B64,
                },
            )
    assert response.status_code == 200
    assert response.json()["prediction"] == "A"


def test_metrics_endpoint_exposes_prometheus() -> None:
    """/metrics exposes Prometheus metrics when the instrumentator is installed (M28).

    Skips when prometheus-fastapi-instrumentator isn't present (e.g. CI's --dev),
    matching the optional-deployed-dep pattern used by the drift tests.
    """
    pytest.importorskip("prometheus_fastapi_instrumentator")
    with TestClient(app) as client:
        client.get("/")  # generate at least one request sample
        resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "# HELP" in body and "http_request" in body


def test_predict_emits_parseable_single_line_event(capsys) -> None:
    """The /predict structured event is one JSON line the collector can parse.

    Regression for the live drift-loop break: the event used to go through the
    Rich logger, which wrapped the long line at 80 cols so Cloud Run split it
    into fragmented entries that monitoring.collect could never reassemble. It
    must now be a single-line JSON readable end-to-end by _parse_log_entry ->
    _features_from_log.
    """
    from project_name.monitoring import _features_from_log, _parse_log_entry

    with (
        patch("project_name.api._module", new=object()),
        patch("project_name.api.predict_single", return_value="A"),
    ):
        with TestClient(app) as client:
            client.post(
                "/predict",
                json={
                    "question": "Is water wet?",
                    "choices": ["Yes", "No"],
                    "image_b64": VALID_IMAGE_B64,
                },
            )

    event_lines = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if '"event": "prediction"' in ln
    ]
    assert len(event_lines) == 1, "exactly one single-line prediction event expected"
    payload = _parse_log_entry(event_lines[0])
    assert payload is not None and payload["event"] == "prediction"
    feats = _features_from_log(payload)
    assert feats is not None
    assert feats["num_choices"] == 2
    assert "subject" not in feats


def test_predict_with_hint_and_lecture() -> None:
    """Predict endpoint correctly forwards hint and lecture when provided."""
    with (
        patch("project_name.api._module", new=object()),
        patch("project_name.api.predict_single", return_value="B") as mock_predict,
    ):
        with TestClient(app) as client:
            client.post(
                "/predict",
                json={
                    "question": "What do plants absorb?",
                    "choices": ["Oxygen", "CO2"],
                    "hint": "Think about photosynthesis.",
                    "lecture": "Plants use sunlight to convert CO2.",
                    "image_b64": VALID_IMAGE_B64,
                },
            )
    # Verify hint and lecture were forwarded via prompt_kwargs
    _, kwargs = mock_predict.call_args
    assert kwargs.get("hint") == "Think about photosynthesis."
    assert kwargs.get("lecture") == "Plants use sunlight to convert CO2."


def test_predict_without_hint_and_lecture() -> None:
    """Predict endpoint does not forward hint or lecture when both are empty."""
    with (
        patch("project_name.api._module", new=object()),
        patch("project_name.api.predict_single", return_value="A") as mock_predict,
    ):
        with TestClient(app) as client:
            client.post(
                "/predict",
                json={
                    "question": "Is water wet?",
                    "choices": ["Yes", "No"],
                    "image_b64": VALID_IMAGE_B64,
                },
            )
    _, kwargs = mock_predict.call_args
    assert "hint" not in kwargs
    assert "lecture" not in kwargs


def test_predict_invalid_image_returns_400() -> None:
    """Predict endpoint returns 400 when image_b64 cannot be decoded."""
    with patch("project_name.api._module", new=object()):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "question": "Is water wet?",
                    "choices": ["Yes", "No"],
                    "image_b64": "not-valid-base64!!!",
                },
            )
    assert response.status_code == 400


def test_predict_missing_image_returns_422() -> None:
    """Predict endpoint returns 422 when image_b64 is absent."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"question": "Is water wet?", "choices": ["Yes", "No"]},
        )
    assert response.status_code == 422


def test_monitor_drift_runs_evidently() -> None:
    """Drift endpoint reads reference/current from GCS and returns a verdict."""
    pytest.importorskip("evidently")
    import pandas as pd

    ref = pd.DataFrame(
        {"question_char_len": [10, 12, 11, 13] * 5, "num_choices": [4] * 20}
    )
    cur = pd.DataFrame(
        {"question_char_len": [9, 12, 10, 14] * 5, "num_choices": [4] * 20}
    )
    with patch("project_name.api._read_gcs_csv", side_effect=[ref, cur]):
        with TestClient(app) as client:
            response = client.get("/monitor/drift")
    assert response.status_code == 200
    body = response.json()
    assert body["n_columns"] == 2
    assert body["reference_rows"] == 20 and body["current_rows"] == 20
    assert isinstance(body["dataset_drift"], bool)
    # With no override and a non-empty production table, it compares against
    # the collected production inputs, not the demo self-comparison sample.
    assert body["current_source"].endswith("current_production.csv")


def test_monitor_drift_handles_failure() -> None:
    """Drift endpoint returns 500 when the source read fails."""
    pytest.importorskip("evidently")
    with patch("project_name.api._read_gcs_csv", side_effect=RuntimeError("no gcs")):
        with TestClient(app) as client:
            response = client.get("/monitor/drift")
    assert response.status_code == 500


class TestResolveCurrent:
    """Unit tests for the current-distribution fallback chain (#4)."""

    def test_explicit_override_wins(self) -> None:
        """An explicit current_gcs is read directly and reported as the source."""
        import pandas as pd

        from project_name import api

        with patch(
            "project_name.api._read_gcs_csv", return_value=pd.DataFrame({"x": [1]})
        ) as mock_read:
            df, source = api._resolve_current("gs://b/explicit.csv")
        assert source == "gs://b/explicit.csv"
        assert len(df) == 1
        mock_read.assert_called_once_with("gs://b/explicit.csv")

    def test_prefers_nonempty_production(self) -> None:
        """With no override, a non-empty production table is used."""
        import pandas as pd

        from project_name import api

        with patch(
            "project_name.api._read_gcs_csv", return_value=pd.DataFrame({"x": [1, 2]})
        ):
            _, source = api._resolve_current(None)
        assert source == api.PRODUCTION_GCS

    def test_falls_back_when_production_missing(self) -> None:
        """A missing/erroring production table falls back to the demo sample."""
        import pandas as pd

        from project_name import api

        def fake(uri: str):
            if uri == api.PRODUCTION_GCS:
                raise FileNotFoundError("no production data yet")
            return pd.DataFrame({"x": [1]})

        with patch("project_name.api._read_gcs_csv", side_effect=fake):
            _, source = api._resolve_current(None)
        assert source == api.CURRENT_SAMPLE_GCS

    def test_falls_back_when_production_empty(self) -> None:
        """An empty production table also falls back to the demo sample."""
        import pandas as pd

        from project_name import api

        def fake(uri: str):
            if uri == api.PRODUCTION_GCS:
                return pd.DataFrame({"x": []})
            return pd.DataFrame({"x": [1]})

        with patch("project_name.api._read_gcs_csv", side_effect=fake):
            _, source = api._resolve_current(None)
        assert source == api.CURRENT_SAMPLE_GCS
