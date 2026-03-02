"""Shared fixtures for all unit tests in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import pytest


@pytest.fixture
def fake_diagnosis() -> dict:
    """Return a fake ECG diagnosis result matching the inference output schema.

    Returns:
        dict: Diagnosis dict with all expected keys.
    """
    return {
        "class": "NORM",
        "class_idx": 0,
        "probability": 0.9,
        "all_probabilities": {"NORM": 0.9, "MI": 0.025, "STTC": 0.025, "CD": 0.025, "HYP": 0.025},
        "is_dangerous": False,
        "description": "Normal sinus rhythm",
    }
