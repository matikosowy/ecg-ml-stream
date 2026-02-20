"""Helper functions & classes module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import numpy as np


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Apply per-channel Z-score normalization to a signal array.

    Args:
        signal (np.ndarray): Array of shape (channels, samples).

    Returns:
        np.ndarray:Normalized array of the same shape.

    """
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True) + 1e-8 # epsilon to prevent division by zero
    return (signal - mean) / std
