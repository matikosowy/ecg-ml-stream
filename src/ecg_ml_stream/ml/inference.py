"""Inference module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from pathlib import Path

import numpy as np
import torch

from ecg_ml_stream.config import cfg
from ecg_ml_stream.ml.model import ECGClassifier
from ecg_ml_stream.utils.helpers import create_sliding_windows, setup_logging

logger = setup_logging(name="inference")


class _ModelCache:
    """Simple cache for lazy-loading the ECG classification model."""

    instance: ECGClassifier | None = None
    path: str | None = None


_cache = _ModelCache()


def get_model(model_path: str | None = None) -> ECGClassifier:
    """Return a ECGClassifier, loading it from disk on first call.

    The model is laoded once per Spark partition. Following calls
    will return the same cached instance.

    Args:
        model_path (str): Path to the ECG classification model.

    Returns:
        ECGClassifier: Loaded ECG classification model.

    """
    model_path = model_path or cfg.model.path

    if _cache.instance is None or _cache.path != model_path:
        _cache.path = model_path

        if Path(model_path).exists():
            _cache.instance = ECGClassifier(model_path=model_path)
            logger.info("Model loaded from %s", model_path)
        else:
            _cache.instance = ECGClassifier()
            logger.warning("Model not found at %s. Using random weights.", model_path)

    return _cache.instance


def infer_ecg_record(
    signal_data: list[list[float]],
    sampling_rate: int | None = None,
    model_path: str | None = None,
) -> dict[str, any]:
    """Run full inference pipeline on a single ECG record.

    Convert raw signal data to overlapping windows, normalize them
    and aggregate predictions with voting.

    Args:
        signal_data (list[list[float]]): Raw ECG signal data as list of channels.
        sampling_rate (int): Sampling rate of the ECG signal.
        model_path (str): Path to the ECG classification model.

    Returns:
        dict: Inference results containing predicted label and confidence.

    """
    sampling_rate = sampling_rate if sampling_rate is not None else cfg.data.sampling_rate
    model_path = model_path or cfg.model.path

    signal = np.stack([np.asarray(ch, dtype=np.float32) for ch in signal_data])

    window_size = int(cfg.data.window_size_sec * sampling_rate)
    stride = int(cfg.data.window_stride_sec * sampling_rate)

    windows = create_sliding_windows(signal, window_size, stride, normalize=True)

    classifier = get_model(model_path)
    windows_tensor = torch.from_numpy(windows).float()
    return classifier.predict_windows(windows_tensor)
