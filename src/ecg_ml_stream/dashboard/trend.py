"""Patient history tracking module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from collections import deque

import numpy as np

from ecg_ml_stream.utils.constants import NUM_LEADS


def extract_signal_features(signal_data: list[list[float]]) -> np.ndarray | None:
    """Extract per-lead statistics from raw ECG signal data.

    For each lead computes:
        - mean amplitude (mean of absolute values)
        - standard deviation

    Args:
        signal_data: Nested list of shape (num_leads, num_samples).

    Returns:
        Numpy array of shape (num_leads * 2,) or None if input is invalid.

    """
    if not signal_data or not signal_data[0]:
        return None

    features = []
    for lead in signal_data[:NUM_LEADS]:
        arr = np.asarray(lead, dtype=np.float64)
        features.extend((np.mean(np.abs(arr)), np.std(arr)))

    return np.array(features, dtype=np.float64)


class PatientHistoryTracker:
    """Track examination history per patient and compute inter-exam comparison metrics.

    For each new diagnosis the tracker returns:
        - which sequential exam this is for the patient
        - Euclidean distance between signal features of the current and previous exam
        - whether the diagnosis class changed since the previous exam
    """

    def __init__(
        self,
        max_exams_per_patient: int = 50,
        n_leads: int = NUM_LEADS,
    ) -> None:
        """Initialize the tracker.

        Args:
            max_exams_per_patient: Maximum exams stored per patient (FIFO).
            n_leads: Number of ECG leads used for feature extraction.

        """
        self._max_exams = max_exams_per_patient
        self._n_leads = n_leads
        self._history: dict[int, deque] = {}

    def update(self, diagnosis: dict) -> dict:
        """Register a new exam and return comparison with the previous one.

        Args:
            diagnosis: Parsed diagnosis dict.  Must contain 'patient_id',
                'diagnosis_class', 'signal_data', 'exam_id', and
                'timestamp_processed'.

        Returns:
            Dict with keys:
                - exam_number (int | None): 1-based position in patient history.
                - feature_deviation (float | None): Euclidean distance between
                  the 24-feature vector of this exam and the previous one.
                  None for the first exam or when signal data is unavailable.
                - class_changed (bool | None): True when the diagnosis class
                  differs from the previous exam.  None for the first exam.
                - prev_diagnosis_class (str | None): Class from the previous exam.

        """
        result: dict = {
            "exam_number": None,
            "feature_deviation": None,
            "class_changed": None,
            "prev_diagnosis_class": None,
        }

        patient_id = diagnosis.get("patient_id")
        if patient_id is None:
            return result

        if patient_id not in self._history:
            self._history[patient_id] = deque(maxlen=self._max_exams)

        history = self._history[patient_id]
        result["exam_number"] = len(history) + 1

        features = extract_signal_features(diagnosis.get("signal_data"))

        if history:
            prev = history[-1]
            result["prev_diagnosis_class"] = prev["diagnosis_class"]
            result["class_changed"] = prev["diagnosis_class"] != diagnosis.get("diagnosis_class")
            prev_features = prev.get("features")
            if features is not None and prev_features is not None:
                result["feature_deviation"] = float(np.linalg.norm(features - prev_features))

        history.append({
            "exam_id": diagnosis.get("exam_id"),
            "timestamp_processed": diagnosis.get("timestamp_processed"),
            "diagnosis_class": diagnosis.get("diagnosis_class"),
            "features": features,
        })

        return result

    def get_patient_history(self, patient_id: int) -> list[dict]:
        """Return all stored exams for a patient (oldest first).

        Args:
            patient_id: PTB-XL patient identifier.

        Returns:
            List of exam dicts with exam_id, timestamp_processed, and
            diagnosis_class fields.

        """
        return [
            {k: v for k, v in entry.items() if k != "features"}
            for entry in self._history.get(patient_id, [])
        ]

    def get_stats(self) -> dict:
        """Return summary statistics across all tracked patients.

        Returns:
            Dict with total_patients and patients_with_history counts.

        """
        total = len(self._history)
        with_history = sum(1 for h in self._history.values() if len(h) > 1)
        return {
            "total_patients": total,
            "patients_with_history": with_history,
        }
