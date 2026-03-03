"""Helper functions & classes module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Apply per-channel Z-score normalization to a signal array.

    Args:
        signal (np.ndarray): Array of shape (channels, samples).

    Returns:
        np.ndarray:Normalized array of the same shape.

    """
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True) + 1e-8  # epsilon to prevent division by zero
    return (signal - mean) / std


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.

    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "logs", name: str = "training") -> logging.Logger:
    """Configure file and console logging.

    Args:
        log_dir: Directory where log files will be written.
        name: Logger name (also used as filename prefix).

    Returns:
        Configured Logger instance.

    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005 - No timezone needed
    log_file = log_path / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False

    return logger


def get_device() -> torch.device:
    """Return the best available compute device.

    Returns:
        torch.device: CUDA if available, then MPS (Apple Silicon), then CPU.

    """
    logger = logging.getLogger(__name__)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch module to inspect.

    Returns:
        Total number of trainable parameters.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
    is_best: bool = False,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        model: The PyTorch model whose state dict will be saved.
        optimizer: Optimizer whose state dict will be saved.
        epoch: Current training epoch (0-indexed).
        metrics: Dictionary of metric values to store alongside the checkpoint.
        path: Destination file path (parent directories are created if needed).
        is_best: If True, also save a copy as ``best_model.pt`` in the same directory.

    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),  # noqa: DTZ005 - No timezone needed
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.parent / "best_model.pt"
        torch.save(checkpoint, best_path)


def create_sliding_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    normalize: bool = False,
) -> np.ndarray:
    """Extract overlapping windows from an ECG signal.

    Args:
        signal (np.ndarray): Input signal of shape (channels, samples).
        window_size (int): Number of samples in each window.
        stride (int): Number of samples to move between windows.
        normalize (bool): Whether to apply per-channel normalization.

    Returns:
        np.ndarray: Array of shape (num_windows, channels, window_size).

    """
    num_channels, signal_length = signal.shape
    windows = []

    start = 0
    while start + window_size <= signal_length:
        window = signal[:, start : start + window_size]
        if normalize:
            window = normalize_signal(window)
        windows.append(window)
        start += stride

    if not windows:
        return np.empty((0, num_channels, window_size), dtype=signal.dtype)

    return np.stack(windows, axis=0)
