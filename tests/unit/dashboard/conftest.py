"""Fixtures for dashboard unit tests in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from collections.abc import Callable

import pytest


@pytest.fixture
def fake_message() -> dict:
    """Return a fake Kafka diagnosis message matching the auxiliary module's expected format.

    Returns:
        dict: Message dict with all fields used by parse_diagnosis_message.
    """
    return {
        "exam_id": "abc-123",
        "timestamp_sent": "2026-01-01T00:00:00",
        "timestamp_processed": "2026-01-01T00:00:01",
        "hospital": {"id": "H1", "name": "Hospital A", "city": "Warsaw"},
        "patient": {"patient_id": 13619, "age": 45, "sex": "M", "ecg_id": 42},
        "diagnosis_class": "NORM",
        "diagnosis_probability": 0.9,
        "all_probabilities": {"NORM": 0.9, "MI": 0.05, "STTC": 0.02, "CD": 0.02, "HYP": 0.01},
        "is_dangerous": False,
        "diagnosis_description": "Normal sinus rhythm",
        "processing_time_ms": 100.0,
        "metadata": {"ground_truth_name": "NORM"},
        "signal_data": [[0.1, 0.2, 0.3]],
        "sampling_rate": 100,
    }


@pytest.fixture
def make_signal() -> Callable[..., list[list[float]]]:
    """Return a factory that creates a synthetic multi-lead ECG signal as nested lists.

    Returns:
        Callable: Factory accepting num_leads and num_samples keyword arguments.
    """
    def _factory(num_leads: int = 12, num_samples: int = 300) -> list[list[float]]:
        return [[float(i % 10) for i in range(num_samples)] for _ in range(num_leads)]

    return _factory
