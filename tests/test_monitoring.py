"""Tests for the data-drift monitoring helpers (M27)."""

from pathlib import Path

from scipali.monitoring import monitoring
from scipali.monitoring.monitoring import _features_from_log, _parse_log_entry


class TestSeedReference:
    """Regenerate the reference + demo-sample drift tables in GCS (repro gap)."""

    def test_writes_both_tables_with_features(self, monkeypatch) -> None:
        """Both tables are written from the right splits via the shared schema."""
        fake = {
            "train": [
                {
                    "question": "What is the capital?",
                    "choices": ["a", "b"],
                    "hint": "h",
                    "lecture": "",
                    "image": None,
                    "subject": "social",
                }
            ],
            "test": [
                {"question": "x", "choices": ["a", "b", "c"], "image": None},
            ],
        }
        monkeypatch.setattr(monitoring, "load_from_disk", lambda _p: fake)
        written: dict = {}
        monkeypatch.setattr(
            monitoring,
            "_write_gcs_csv",
            lambda df, uri: written.__setitem__(uri, df),
        )

        monitoring.seed_reference(
            processed_dir=Path("ignored"),
            reference="train",
            sample="test",
            reference_gcs="gs://b/reference.csv",
            sample_gcs="gs://b/current_sample.csv",
        )

        assert set(written) == {"gs://b/reference.csv", "gs://b/current_sample.csv"}
        ref = written["gs://b/reference.csv"]
        assert len(ref) == 1 and "subject" in ref.columns
        assert ref.iloc[0]["hint_present"] == 1
        assert len(written["gs://b/current_sample.csv"]) == 1


class TestFeaturesFromLog:
    """Map a /predict log event to the reference feature schema (#5)."""

    def test_maps_all_fields(self) -> None:
        """Each logged field maps to the matching reference feature."""
        event = {
            "event": "prediction",
            "question": "What is 2+2?",
            "n_choices": 4,
            "hint": True,
            "lecture": False,
            "image_px": [320, 240],
            "prediction": "A",
        }
        assert _features_from_log(event) == {
            "question_char_len": 12,
            "question_word_len": 3,
            "num_choices": 4,
            "hint_present": 1,
            "lecture_present": 0,
            "image_width": 320,
            "image_height": 240,
        }

    def test_omits_subject(self) -> None:
        """Subject is unknown at inference time, so it must not be emitted."""
        feats = _features_from_log(
            {"event": "prediction", "question": "x", "n_choices": 2, "image_px": [1, 1]}
        )
        assert feats is not None and "subject" not in feats

    def test_non_prediction_returns_none(self) -> None:
        """Non-prediction events are skipped."""
        assert _features_from_log({"event": "startup"}) is None

    def test_missing_image_px_defaults_zero(self) -> None:
        """A missing image_px does not raise and yields zero dimensions."""
        feats = _features_from_log(
            {"event": "prediction", "question": "x", "n_choices": 2}
        )
        assert feats is not None
        assert feats["image_width"] == 0 and feats["image_height"] == 0


class TestParseLogEntry:
    """Defensively pull the JSON event from a Cloud Logging payload."""

    def test_text_payload(self) -> None:
        """A 'prediction {json}' text payload is parsed."""
        out = _parse_log_entry('prediction {"event": "prediction", "n_choices": 3}')
        assert out == {"event": "prediction", "n_choices": 3}

    def test_text_payload_with_internal_newlines(self) -> None:
        """Soft-wrap newlines inside the JSON are tolerated."""
        out = _parse_log_entry('prediction {"event":\n "prediction"}')
        assert out == {"event": "prediction"}

    def test_structured_dict_payload(self) -> None:
        """A structured jsonPayload dict with an event is returned as-is."""
        out = _parse_log_entry({"event": "prediction", "n_choices": 3})
        assert out is not None and out["n_choices"] == 3

    def test_dict_without_event_is_none(self) -> None:
        """A dict lacking 'event' is not a prediction line."""
        assert _parse_log_entry({"foo": "bar"}) is None

    def test_text_without_json_is_none(self) -> None:
        """Plain log lines with no JSON object yield None."""
        assert _parse_log_entry("Model ready.") is None

    def test_non_string_non_dict_is_none(self) -> None:
        """Unexpected payload types yield None rather than raising."""
        assert _parse_log_entry(12345) is None

    def test_rich_truncated_fragment_is_none(self) -> None:
        """A Rich-wrapped 80-col fragment (the old live bug) is rejected.

        Rich rendered the event at a fixed 80 columns with a right-aligned
        source suffix, so Cloud Run stored a truncated, invalid-JSON first line
        like the one below (the rest landed in separate entries). It must parse
        to None, not raise — the fix emits a single-line JSON instead.
        """
        fragment = (
            '[06/15/26 12:26:28] INFO     prediction {"event": "prediction",'
            "       api.py:302"
        )
        assert _parse_log_entry(fragment) is None
