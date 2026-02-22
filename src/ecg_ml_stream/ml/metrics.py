"""Training metrics module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ecg_ml_stream.utils.constants import CLASS_NAMES


class MetricsCalculator:
    """Accumulate predictions and compute multi-class classification metrics.

    Attributes:
        CLASS_NAMES (ClassVar[list[str]]): List of ECG class names.

    """

    CLASS_NAMES: ClassVar[list[str]] = CLASS_NAMES

    def __init__(self) -> None:
        """Initialize MetricsCalculator with empty state."""
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated predictions and losses."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: float | None = None,
    ) -> None:
        """Add a batch predictions to the accumulator.

        Args:
            preds (torch.Tensor): Model output logits for the batch.
            targets (torch.Tensor): Ground truth labels for the batch.
            loss (float | None): Optional loss value for the batch.

        """
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)

        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> dict:
        """Compute all accumulated metrics.

        Returns:
            dict: Dictionary containing accuracy, precision, recall,
            F1 (macro/weighted), per-class F1, and mean loss.

        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "precision_macro": precision_score(
                targets,
                preds,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                targets,
                preds,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                targets,
                preds,
                average="macro",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                targets,
                preds,
                average="weighted",
                zero_division=0,
            ),
        }

        f1_per_class = f1_score(
            targets, preds, average=None, zero_division=0, labels=list(range(len(self.CLASS_NAMES)))
        )
        for idx, class_name in enumerate(self.CLASS_NAMES):
            metrics[f"f1_{class_name}"] = f1_per_class[idx]

        if self.losses:
            metrics["loss"] = np.mean(self.losses)

        return metrics


class AverageMeter:
    """Track the running average of a scalar metric."""

    def __init__(self) -> None:
        """Initialize AverageMeter with empty state."""
        self.reset()

    def reset(self) -> None:
        """Reset all tracked values to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add a new observation to the running average.

        Args:
            val (float): Observed value to add.
            n (int): Weight of the observation (e.g., batch size).

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_metrics(metrics: dict) -> str:
    """Format a metrics dictionary into a readable string.

    Args:
        metrics (dict): Dictionary of metric names and values.

    Returns:
        str: Formatted string of metrics.

    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def save_training_history(history: dict, path: str) -> None:
    """Save training history to a JSON file.

    Args:
        history (dict): Dictionary containing training history, mapping names
            to list of per-epoch values.
        path (str): File path to save the history JSON.

    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_training_history(path: str) -> dict:
    """Load training history from a JSON file.

    Args:
        path (str): File path to load the history JSON from.

    Returns:
        dict: Dictionary containing training history, mapping names to list of per-epoch values.

    """
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: "min" for loss, "max" for accuracy/F1.

        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Update internal state and return whether training should stop.

        Args:
            score: Current epoch metric value.

        Returns:
            True if training should stop, False otherwise.

        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
