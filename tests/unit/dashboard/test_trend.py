"""Unit tests for TrendTracker in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import numpy as np
import pytest

from ecg_ml_stream.dashboard.trend import TrendTracker, extract_signal_features
from ecg_ml_stream.utils.constants import CLASS_NAMES, NUM_LEADS

_N_FEATURES = NUM_LEADS * 2


def _make_signal(num_leads: int = NUM_LEADS, num_samples: int = 100) -> list[list[float]]:
    rng = np.random.default_rng(42)
    return [rng.normal(0.0, 1.0, size=num_samples).tolist() for _ in range(num_leads)]


def _make_diagnosis(
    cls: str = "NORM",
    signal: list[list[float]] | None = None,
    exam_id: str = "exam-1",
    timestamp: str = "2026-01-01T00:00:00",
) -> dict:
    if signal is None:
        signal = _make_signal()
    return {
        "diagnosis_class": cls,
        "signal_data": signal,
        "exam_id": exam_id,
        "timestamp_processed": timestamp,
    }


class TestExtractSignalFeatures:
    def test_returns_correct_shape(self):
        signal = _make_signal()
        features = extract_signal_features(signal)
        assert features.shape == (_N_FEATURES,)

    def test_returns_none_for_empty_input(self):
        assert extract_signal_features([]) is None
        assert extract_signal_features([[]]) is None

    def test_mean_amplitude_is_mean_of_abs(self):
        signal = [[1.0, -2.0, 3.0]] + [[0.0] * 3] * (NUM_LEADS - 1)
        features = extract_signal_features(signal)
        expected_amp = np.mean(np.abs([1.0, -2.0, 3.0]))
        assert features[0] == pytest.approx(expected_amp)

    def test_std_matches_numpy(self):
        vals = [1.0, -2.0, 3.0, 0.5]
        signal = [vals] + [[0.0] * len(vals)] * (NUM_LEADS - 1)
        features = extract_signal_features(signal)
        assert features[1] == pytest.approx(np.std(vals))


class TestTrendTrackerInit:
    @pytest.fixture
    def tracker(self) -> TrendTracker:
        return TrendTracker(CLASS_NAMES)

    def test_count_is_zero_for_all_classes(self, tracker):
        for c in CLASS_NAMES:
            assert tracker._count[c] == 0

    def test_centroid_is_none_for_all_classes(self, tracker):
        for c in CLASS_NAMES:
            assert tracker.get_centroid(c) is None

    def test_deviation_history_is_empty(self, tracker):
        assert tracker.get_deviation_history() == []


class TestTrendTrackerUpdate:
    @pytest.fixture
    def tracker(self) -> TrendTracker:
        return TrendTracker(CLASS_NAMES)

    def test_single_update_increments_count(self, tracker):
        tracker.update(_make_diagnosis())
        assert tracker._count["NORM"] == 1

    def test_single_update_sets_centroid(self, tracker):
        diag = _make_diagnosis()
        tracker.update(diag)
        centroid = tracker.get_centroid("NORM")
        assert centroid is not None
        assert centroid.shape == (_N_FEATURES,)

    def test_centroid_matches_features_after_one_sample(self, tracker):
        signal = _make_signal()
        diag = _make_diagnosis(signal=signal)
        tracker.update(diag)
        centroid = tracker.get_centroid("NORM")
        expected = extract_signal_features(signal)
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_two_updates_produce_averaged_centroid(self, tracker):
        sig_a = _make_signal(num_samples=50)
        sig_b = _make_signal(num_samples=80)

        tracker.update(_make_diagnosis(signal=sig_a))
        tracker.update(_make_diagnosis(signal=sig_b, exam_id="exam-2"))

        centroid = tracker.get_centroid("NORM")
        feat_a = extract_signal_features(sig_a)
        feat_b = extract_signal_features(sig_b)
        expected = (feat_a + feat_b) / 2
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_returns_float_deviation(self, tracker):
        result = tracker.update(_make_diagnosis())
        assert isinstance(result, float)

    def test_returns_none_for_missing_signal(self, tracker):
        diag = {"diagnosis_class": "NORM", "signal_data": None}
        assert tracker.update(diag) is None

    def test_returns_none_for_empty_signal(self, tracker):
        diag = {"diagnosis_class": "NORM", "signal_data": []}
        assert tracker.update(diag) is None

    def test_returns_none_for_unknown_class(self, tracker):
        diag = _make_diagnosis(cls="UNKNOWN")
        assert tracker.update(diag) is None

    def test_variance_is_none_after_one_sample(self, tracker):
        tracker.update(_make_diagnosis())
        assert tracker.get_variance("NORM") is None

    def test_variance_shape_after_two_samples(self, tracker):
        tracker.update(_make_diagnosis(exam_id="e1"))
        tracker.update(_make_diagnosis(exam_id="e2"))
        variance = tracker.get_variance("NORM")
        assert variance is not None
        assert variance.shape == (_N_FEATURES,)

    def test_history_appended(self, tracker):
        tracker.update(_make_diagnosis())
        history = tracker.get_deviation_history("NORM")
        assert len(history) == 1
        assert history[0]["exam_id"] == "exam-1"


class TestTrendTrackerDeviation:
    @pytest.fixture
    def tracker(self) -> TrendTracker:
        return TrendTracker(CLASS_NAMES)

    def test_first_sample_deviation_is_zero(self, tracker):
        deviation = tracker.update(_make_diagnosis())
        assert deviation == pytest.approx(0.0)

    def test_identical_samples_have_zero_deviation(self, tracker):
        signal = _make_signal()
        tracker.update(_make_diagnosis(signal=signal, exam_id="e1"))
        dev = tracker.update(_make_diagnosis(signal=signal, exam_id="e2"))
        assert dev == pytest.approx(0.0)

    def test_different_signal_has_nonzero_deviation(self, tracker):
        rng = np.random.default_rng(0)
        sig_a = [rng.normal(0, 1, 100).tolist() for _ in range(NUM_LEADS)]
        sig_b = [rng.normal(5, 2, 100).tolist() for _ in range(NUM_LEADS)]

        tracker.update(_make_diagnosis(signal=sig_a))
        dev = tracker.update(_make_diagnosis(signal=sig_b, exam_id="e2"))
        assert dev > 0.0


class TestTrendTrackerHistory:
    @pytest.fixture
    def tracker(self) -> TrendTracker:
        return TrendTracker(CLASS_NAMES)

    def test_empty_initially(self, tracker):
        assert tracker.get_deviation_history() == []
        assert tracker.get_deviation_history("NORM") == []

    def test_respects_max_history(self):
        tracker = TrendTracker(CLASS_NAMES, max_history=3)
        for i in range(5):
            tracker.update(_make_diagnosis(exam_id=f"exam-{i}"))

        history = tracker.get_deviation_history("NORM")
        assert len(history) == 3
        assert history[0]["exam_id"] == "exam-2"

    def test_entry_structure(self, tracker):
        tracker.update(_make_diagnosis())
        entry = tracker.get_deviation_history("NORM")[0]
        assert "timestamp" in entry
        assert "deviation" in entry
        assert "exam_id" in entry
        assert "class_name" in entry

    def test_combined_history_returns_all_classes(self, tracker):
        tracker.update(_make_diagnosis(cls="NORM"))
        tracker.update(
            _make_diagnosis(cls="MI", exam_id="exam-2", timestamp="2026-01-01T00:00:01")
        )

        combined = tracker.get_deviation_history()
        assert len(combined) == 2
        classes = {e["class_name"] for e in combined}
        assert classes == {"NORM", "MI"}

    def test_combined_history_sorted_by_timestamp(self, tracker):
        tracker.update(_make_diagnosis(cls="MI", timestamp="2026-01-01T00:00:02"))
        tracker.update(
            _make_diagnosis(cls="NORM", timestamp="2026-01-01T00:00:01", exam_id="exam-2")
        )

        combined = tracker.get_deviation_history()
        assert combined[0]["timestamp"] < combined[1]["timestamp"]


class TestGetCentroidPerLead:
    def test_returns_none_when_no_data(self):
        tracker = TrendTracker(CLASS_NAMES)
        assert tracker.get_centroid_per_lead("NORM") is None

    def test_returns_dict_with_lead_keys(self):
        tracker = TrendTracker(CLASS_NAMES)
        tracker.update(_make_diagnosis())
        result = tracker.get_centroid_per_lead("NORM")
        assert len(result) == NUM_LEADS
        assert all(isinstance(result[i], dict) for i in range(NUM_LEADS))

    def test_per_lead_contains_expected_keys(self):
        tracker = TrendTracker(CLASS_NAMES)
        tracker.update(_make_diagnosis())
        result = tracker.get_centroid_per_lead("NORM")
        for i in range(NUM_LEADS):
            assert "mean_amplitude" in result[i]
            assert "std" in result[i]


class TestGetClassStats:
    @pytest.fixture
    def tracker(self) -> TrendTracker:
        return TrendTracker(CLASS_NAMES)

    def test_returns_all_classes(self, tracker):
        stats = tracker.get_class_stats()
        assert set(stats.keys()) == set(CLASS_NAMES)

    def test_count_reflects_updates(self, tracker):
        tracker.update(_make_diagnosis(cls="NORM"))
        tracker.update(_make_diagnosis(cls="NORM", exam_id="exam-2"))
        tracker.update(_make_diagnosis(cls="MI", exam_id="exam-3"))

        stats = tracker.get_class_stats()
        assert stats["NORM"]["count"] == 2
        assert stats["MI"]["count"] == 1
        assert stats["STTC"]["count"] == 0
