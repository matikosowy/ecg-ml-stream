"""Fixtures for producer unit tests in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import pytest

from ecg.ml.model import ECGClassifier, ResNet1D


@pytest.fixture
def untrained_model() -> ResNet1D:
    """Return a freshly initialized ResNet1D with random weights.

    Returns:
        ResNet1D instance (12 leads, 5 classes).
    """
    return ResNet1D(input_channels=12, num_classes=5)


@pytest.fixture
def untrained_classifier() -> ECGClassifier:
    """Return an ECGClassifier with random weights (no checkpoint loaded).

    Returns:
        ECGClassifier instance on CPU.
    """
    return ECGClassifier()
