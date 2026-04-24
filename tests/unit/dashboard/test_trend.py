"""Unit tests for PatientHistoryTracker in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import numpy as np
import pytest

from ecg_ml_stream.dashboard.trend import PatientHistoryTracker, extract_signal_features
from ecg_ml_stream.utils.constants import NUM_LEADS

_N_FEATURES = NUM_LEADS * 2


def _make_signal(num_leads: int = NUM_LEADS, num_samples: int = 100) -> list[list[float]]:
    rng = np.random.default_rng(42)
    return [rng.normal(0.0, 1.0, size=num_samples).tolist() for _ in range(num_leads)]


def _make_diagnosis(
    patient_id: int | None = 1,
    cls: str = "NORM",
    signal: list[list[float]] | None = None,
    exam_id: str = "exam-1",
    timestamp: str = "2026-01-01T00:00:00",
) -> dict:
    if signal is None:
        signal = _make_signal()
    return {
        "patient_id": patient_id,
        "diagnosis_class": cls,
        "signal_data": signal,
        "exam_id": exam_id,
        "timestamp_processed": timestamp,
    }


class TestExtractSignalFeatures:
    def test_returns_correct_shape(self):
        features = extract_signal_features(_make_signal())
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


class TestPatientHistoryTrackerInit:
    def test_stats_are_zero_initially(self):
        tracker = PatientHistoryTracker()
        stats = tracker.get_stats()
        assert stats["total_patients"] == 0
        assert stats["patients_with_history"] == 0

    def test_patient_history_empty_initially(self):
        tracker = PatientHistoryTracker()
        assert tracker.get_patient_history(99) == []


class TestPatientHistoryTrackerUpdate:
    @pytest.fixture
    def tracker(self) -> PatientHistoryTracker:
        return PatientHistoryTracker()

    def test_returns_dict_with_required_keys(self, tracker):
        result = tracker.update(_make_diagnosis())
        assert {"exam_number", "feature_deviation", "class_changed", "prev_diagnosis_class"} <= set(
            result
        )

    def test_first_exam_has_number_one(self, tracker):
        result = tracker.update(_make_diagnosis())
        assert result["exam_number"] == 1

    def test_second_exam_has_number_two(self, tracker):
        tracker.update(_make_diagnosis(exam_id="e1"))
        result = tracker.update(_make_diagnosis(exam_id="e2"))
        assert result["exam_number"] == 2

    def test_first_exam_has_no_comparison(self, tracker):
        result = tracker.update(_make_diagnosis())
        assert result["feature_deviation"] is None
        assert result["class_changed"] is None
        assert result["prev_diagnosis_class"] is None

    def test_second_exam_has_feature_deviation(self, tracker):
        tracker.update(_make_diagnosis(exam_id="e1"))
        result = tracker.update(_make_diagnosis(exam_id="e2"))
        assert result["feature_deviation"] is not None
        assert isinstance(result["feature_deviation"], float)

    def test_feature_deviation_is_nonnegative(self, tracker):
        tracker.update(_make_diagnosis(exam_id="e1"))
        result = tracker.update(_make_diagnosis(exam_id="e2"))
        assert result["feature_deviation"] >= 0.0

    def test_identical_signals_have_zero_deviation(self, tracker):
        signal = _make_signal()
        tracker.update(_make_diagnosis(signal=signal, exam_id="e1"))
        result = tracker.update(_make_diagnosis(signal=signal, exam_id="e2"))
        assert result["feature_deviation"] == pytest.approx(0.0)

    def test_different_signals_have_nonzero_deviation(self, tracker):
        rng = np.random.default_rng(0)
        sig_a = [rng.normal(0, 1, 100).tolist() for _ in range(NUM_LEADS)]
        sig_b = [rng.normal(5, 2, 100).tolist() for _ in range(NUM_LEADS)]
        tracker.update(_make_diagnosis(signal=sig_a, exam_id="e1"))
        result = tracker.update(_make_diagnosis(signal=sig_b, exam_id="e2"))
        assert result["feature_deviation"] > 0.0

    def test_class_changed_is_false_for_same_class(self, tracker):
        tracker.update(_make_diagnosis(cls="NORM", exam_id="e1"))
        result = tracker.update(_make_diagnosis(cls="NORM", exam_id="e2"))
        assert result["class_changed"] is False

    def test_class_changed_is_true_when_class_differs(self, tracker):
        tracker.update(_make_diagnosis(cls="NORM", exam_id="e1"))
        result = tracker.update(_make_diagnosis(cls="MI", exam_id="e2"))
        assert result["class_changed"] is True

    def test_prev_diagnosis_class_matches_previous(self, tracker):
        tracker.update(_make_diagnosis(cls="STTC", exam_id="e1"))
        result = tracker.update(_make_diagnosis(cls="NORM", exam_id="e2"))
        assert result["prev_diagnosis_class"] == "STTC"

    def test_no_patient_id_returns_none_fields(self, tracker):
        result = tracker.update(_make_diagnosis(patient_id=None))
        assert result["exam_number"] is None
        assert result["feature_deviation"] is None
        assert result["class_changed"] is None

    def test_missing_signal_skips_deviation(self, tracker):
        tracker.update(_make_diagnosis(exam_id="e1"))
        diag = _make_diagnosis(exam_id="e2")
        diag["signal_data"] = None
        result = tracker.update(diag)
        assert result["feature_deviation"] is None

    def test_different_patients_tracked_independently(self, tracker):
        tracker.update(_make_diagnosis(patient_id=1, cls="NORM", exam_id="e1"))
        tracker.update(_make_diagnosis(patient_id=2, cls="MI", exam_id="e2"))
        result_p1 = tracker.update(_make_diagnosis(patient_id=1, cls="MI", exam_id="e3"))
        assert result_p1["exam_number"] == 2
        assert result_p1["prev_diagnosis_class"] == "NORM"

    def test_stats_update_after_updates(self, tracker):
        tracker.update(_make_diagnosis(patient_id=1, exam_id="e1"))
        tracker.update(_make_diagnosis(patient_id=1, exam_id="e2"))
        tracker.update(_make_diagnosis(patient_id=2, exam_id="e3"))
        stats = tracker.get_stats()
        assert stats["total_patients"] == 2
        assert stats["patients_with_history"] == 1

    def test_max_exams_per_patient_respected(self):
        tracker = PatientHistoryTracker(max_exams_per_patient=3)
        for i in range(5):
            tracker.update(_make_diagnosis(patient_id=1, exam_id=f"e{i}"))
        history = tracker.get_patient_history(1)
        assert len(history) == 3


class TestGetPatientHistory:
    @pytest.fixture
    def tracker(self) -> PatientHistoryTracker:
        return PatientHistoryTracker()

    def test_returns_list(self, tracker):
        tracker.update(_make_diagnosis())
        assert isinstance(tracker.get_patient_history(1), list)

    def test_entry_has_expected_keys(self, tracker):
        tracker.update(_make_diagnosis())
        entry = tracker.get_patient_history(1)[0]
        assert "exam_id" in entry
        assert "timestamp_processed" in entry
        assert "diagnosis_class" in entry

    def test_features_not_exposed(self, tracker):
        tracker.update(_make_diagnosis())
        entry = tracker.get_patient_history(1)[0]
        assert "features" not in entry

    def test_history_ordered_oldest_first(self, tracker):
        tracker.update(_make_diagnosis(exam_id="first", timestamp="2026-01-01T00:00:00"))
        tracker.update(_make_diagnosis(exam_id="second", timestamp="2026-01-01T00:01:00"))
        history = tracker.get_patient_history(1)
        assert history[0]["exam_id"] == "first"
        assert history[1]["exam_id"] == "second"

    def test_unknown_patient_returns_empty(self, tracker):
        assert tracker.get_patient_history(9999) == []
