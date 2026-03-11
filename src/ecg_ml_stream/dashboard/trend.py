"""Trend tracking module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from collections import deque

import numpy as np

from ecg_ml_stream.utils.constants import CLASS_NAMES, NUM_LEADS

_FEATURES_PER_LEAD = 2  # mean_amplitude, std


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


class TrendTracker:
    """Maintain per-class running statistics on ECG signal features.

    Uses Welford's online algorithm for numerically stable computation
    of running mean and variance. Tracks deviation history for timeline
    visualisation.

    Each sample is represented as a feature vector of per-lead statistics
    (mean amplitude, std) rather than model output probabilities.
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        max_history: int = 200,
        n_leads: int = NUM_LEADS,
    ) -> None:
        """Initialize the trend tracker.

        Args:
            class_names: List of diagnosis class names. Defaults to CLASS_NAMES.
            max_history: Maximum deviation history entries per class.
            n_leads: Number of ECG leads.

        """
        self._class_names = class_names or CLASS_NAMES
        self._n_leads = n_leads
        self._n_features = n_leads * _FEATURES_PER_LEAD
        self._max_history = max_history

        self._count: dict[str, int] = dict.fromkeys(self._class_names, 0)
        self._mean: dict[str, np.ndarray] = {
            c: np.zeros(self._n_features) for c in self._class_names
        }
        self._m2: dict[str, np.ndarray] = {
            c: np.zeros(self._n_features) for c in self._class_names
        }
        self._history: dict[str, deque] = {
            c: deque(maxlen=max_history) for c in self._class_names
        }

    def update(self, diagnosis: dict) -> float | None:
        """Update running statistics with a new diagnosis and return deviation.

        Extracts per-lead signal features, applies Welford's update,
        and computes the z-score based deviation from the class centroid.

        Args:
            diagnosis: Parsed diagnosis dict with 'diagnosis_class',
                'signal_data', 'exam_id', and 'timestamp_processed' keys.

        Returns:
            Deviation score (mean absolute z-score), or None if update skipped.

        """
        class_name = diagnosis.get("diagnosis_class")
        signal_data = diagnosis.get("signal_data")

        if class_name not in self._count:
            return None

        features = extract_signal_features(signal_data)
        if features is None or len(features) != self._n_features:
            return None

        n = self._count[class_name] + 1
        self._count[class_name] = n

        old_mean = self._mean[class_name].copy()
        delta = features - old_mean
        self._mean[class_name] = old_mean + delta / n
        delta2 = features - self._mean[class_name]
        self._m2[class_name] += delta * delta2

        deviation = self._compute_deviation(features, class_name)

        self._history[class_name].append({
            "timestamp": diagnosis.get("timestamp_processed"),
            "deviation": deviation,
            "exam_id": diagnosis.get("exam_id", ""),
            "class_name": class_name,
        })

        return deviation

    def _compute_deviation(
        self,
        features: np.ndarray,
        class_name: str,
    ) -> float:
        """Compute mean absolute z-score of features vs class centroid.

        Returns:
            Mean absolute z-score across all feature dimensions.
            Returns 0.0 if variance is not yet available (< 2 samples).

        """
        n = self._count[class_name]
        if n < 2:  # noqa: PLR2004 - Welford needs at least 2 samples
            return 0.0

        variance = self._m2[class_name] / (n - 1)
        std = np.sqrt(np.maximum(variance, 1e-12))
        z_scores = np.abs(features - self._mean[class_name]) / std
        return float(np.mean(z_scores))

    def get_centroid(self, class_name: str) -> np.ndarray | None:
        """Return the running mean feature vector for a class.

        Returns:
            Numpy array of shape (n_features,) or None if no samples seen.

        """
        if self._count.get(class_name, 0) == 0:
            return None
        return self._mean[class_name].copy()

    def get_centroid_per_lead(self, class_name: str) -> dict[str, dict] | None:
        """Return per-lead centroid statistics for a class.

        Returns:
            Dict mapping lead index to {mean_amplitude, std}, or None.

        """
        centroid = self.get_centroid(class_name)
        if centroid is None:
            return None

        result = {}
        for i in range(self._n_leads):
            base = i * _FEATURES_PER_LEAD
            result[i] = {
                "mean_amplitude": centroid[base],
                "std": centroid[base + 1],
            }
        return result

    def get_variance(self, class_name: str) -> np.ndarray | None:
        """Return the running variance of the feature vector for a class.

        Returns:
            Numpy array of shape (n_features,) or None if fewer than 2 samples.

        """
        n = self._count.get(class_name, 0)
        if n < 2:  # noqa: PLR2004 - Welford needs at least 2 samples
            return None
        return (self._m2[class_name] / (n - 1)).copy()

    def get_class_stats(self) -> dict[str, dict]:
        """Return summary statistics for all classes.

        Returns:
            Dict mapping class name to {count, centroid, variance}.

        """
        return {
            c: {
                "count": self._count[c],
                "centroid": self.get_centroid(c),
                "variance": self.get_variance(c),
            }
            for c in self._class_names
        }

    def get_deviation_history(
        self,
        class_name: str | None = None,
    ) -> list[dict]:
        """Return deviation history entries.

        Args:
            class_name: If provided, return history for that class only.
                If None, return combined history for all classes sorted by timestamp.

        Returns:
            List of dicts with timestamp, deviation, exam_id, class_name.

        """
        if class_name is not None:
            return list(self._history.get(class_name, []))

        combined = []
        for entries in self._history.values():
            combined.extend(entries)
        return sorted(combined, key=lambda x: x.get("timestamp") or "")
