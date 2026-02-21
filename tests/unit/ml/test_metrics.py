"""Unit tests for training metrics module.

Copyright 2026 Mateusz Golebiewski
"""

import pytest
import torch

from ecg.ml.metrics import (
    AverageMeter,
    MetricsCalculator,
    format_metrics,
    load_training_history,
    save_training_history,
)


class TestMetricsCalculator:
    def test_update_with_logits(self):
        calc = MetricsCalculator()
        preds = torch.tensor([[2.0, 0.5, 0.1, 0.0, 0.0]] * 4)
        targets = torch.tensor([0, 0, 0, 0])
        calc.update(preds, targets)
        assert len(calc.predictions) == 4

    def test_update_with_class_indices(self):
        calc = MetricsCalculator()
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])
        calc.update(preds, targets)
        assert len(calc.predictions) == 4

    def test_update_with_loss(self):
        calc = MetricsCalculator()
        preds = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        calc.update(preds, targets, loss=0.42)
        assert calc.losses == [0.42]

    def test_compute_returns_metrics(self):
        calc = MetricsCalculator()
        preds = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        calc.update(preds, targets, loss=0.1)
        metrics = calc.compute()
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "loss" in metrics
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_compute_includes_per_class_f1(self):
        calc = MetricsCalculator()
        preds = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        calc.update(preds, targets)
        metrics = calc.compute()
        assert "f1_NORM" in metrics

    def test_compute_without_loss(self):
        calc = MetricsCalculator()
        preds = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        calc.update(preds, targets)  # no loss provided
        metrics = calc.compute()
        assert "loss" not in metrics


class TestAverageMeter:
    def test_basic(self):
        meter = AverageMeter()
        meter.update(10.0, n=1)
        meter.update(20.0, n=1)
        assert meter.avg == 15.0
        assert meter.count == 2

    def test_weighted(self):
        meter = AverageMeter()
        meter.update(10.0, n=3)
        meter.update(20.0, n=1)
        assert meter.avg == 12.5  # (30 + 20) / 4 = 12.5

    def test_reset(self):
        meter = AverageMeter()
        meter.update(10.0)
        meter.reset()
        assert meter.avg == 0
        assert meter.count == 0

    def test_val_updated(self):
        meter = AverageMeter()
        meter.update(42.0)
        assert meter.val == 42.0


class TestFormatMetrics:
    def test_float_values(self):
        result = format_metrics({"loss": 0.1234, "f1": 0.9876})
        assert "loss: 0.1234" in result
        assert "f1: 0.9876" in result

    def test_non_float_values(self):
        result = format_metrics({"epoch": 5})
        assert "epoch: 5" in result


class TestSaveLoadTrainingHistory:
    def test_roundtrip(self, tmp_path):
        history = {"train_loss": [1.0, 0.9, 0.8], "val_f1": [0.5, 0.6, 0.7]}
        path = str(tmp_path / "history.json")

        save_training_history(history, path)
        loaded = load_training_history(path)

        assert loaded == history

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "history.json")
        save_training_history({"x": [1]}, path)
        assert (tmp_path / "nested" / "dir" / "history.json").exists()
