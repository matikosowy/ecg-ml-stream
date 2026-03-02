"""Unit tests for helpers module.

Copyright 2026 Mateusz Golebiewski
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from ecg_ml_stream.ml.model import ResNet1D
from ecg_ml_stream.utils.helpers import (
    count_parameters,
    create_sliding_windows,
    get_device,
    normalize_signal,
    save_checkpoint,
    set_seed,
    setup_logging,
)


class TestNormalizeSignal:
    def test_zero_mean(self):
        signal = np.random.default_rng(0).standard_normal((12, 250)).astype(np.float32)
        normed = normalize_signal(signal)
        means = normed.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_std(self):
        signal = np.random.default_rng(0).standard_normal((12, 250)).astype(np.float32)
        normed = normalize_signal(signal)
        stds = normed.std(axis=-1)
        np.testing.assert_allclose(stds, 1.0, atol=1e-2)

    def test_constant_channel_no_nan(self):
        signal = np.ones((12, 250), dtype=np.float32)
        normed = normalize_signal(signal)
        assert not np.any(np.isnan(normed))


class TestSetSeed:
    def test_runs_without_error(self):
        set_seed(42)

    def test_different_seeds_give_different_results(self):
        set_seed(0)
        a = torch.randn(10)
        set_seed(1)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


class TestSetupLogging:
    def test_returns_logger(self, tmp_path):
        logger = setup_logging(str(tmp_path), name="test_run")
        assert logger is not None
        assert logger.name == "test_run"


class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_is_valid(self):
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")

    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"),
        [
            (True, True, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        ],
    )
    def test_device_selection(self, cuda, mps, expected):
        with (
            patch("torch.cuda.is_available", return_value=cuda),
            patch("torch.cuda.get_device_name", return_value="Test GPU"),
            patch("torch.backends.mps.is_available", return_value=mps),
        ):
            device = get_device()
        assert device.type == expected


class TestCountParameters:
    def test_positive_count(self):
        model = ResNet1D(input_channels=12, num_classes=5)
        count = count_parameters(model)
        assert count > 0

    def test_zero_params_frozen(self):
        model = ResNet1D(input_channels=12, num_classes=5)
        for p in model.parameters():
            p.requires_grad = False
        assert count_parameters(model) == 0


class TestCreateSlidingWindows:
    def test_returns_correct_number_of_windows(self):
        signal = np.zeros((12, 1000))
        result = create_sliding_windows(signal, window_size=250, stride=125)
        assert result.shape[0] == 7

    def test_window_shape(self):
        signal = np.zeros((12, 1000))
        result = create_sliding_windows(signal, window_size=250, stride=125)
        assert result.shape == (7, 12, 250)

    def test_exact_fit_one_window(self):
        signal = np.zeros((12, 250))
        result = create_sliding_windows(signal, window_size=250, stride=250)
        assert result.shape[0] == 1

    def test_signal_too_short_returns_empty(self):
        signal = np.zeros((12, 100))
        result = create_sliding_windows(signal, window_size=250, stride=125)
        assert result.shape == (0, 12, 250)

    def test_signal_too_short_preserves_dtype(self):
        signal = np.zeros((12, 100), dtype=np.float32)
        result = create_sliding_windows(signal, window_size=250, stride=125)
        assert result.dtype == np.float32

    def test_normalize_true_applies_normalization(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal((12, 500))
        result = create_sliding_windows(signal, window_size=250, stride=250, normalize=True)
        means = result[0].mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_normalize_false_preserves_values(self):
        signal = np.ones((12, 500)) * 5.0
        result = create_sliding_windows(signal, window_size=250, stride=250, normalize=False)
        assert np.all(result == 5.0)


class TestSaveCheckpoint:
    def test_saves_expected_keys(self, tmp_path):
        model = ResNet1D(input_channels=12, num_classes=5)
        optimizer = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "checkpoint.pt")

        save_checkpoint(model, optimizer, epoch=3, metrics={"f1": 0.85}, path=path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 3
        assert checkpoint["metrics"]["f1"] == pytest.approx(0.85)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_save_is_best_creates_copy(self, tmp_path):
        model = ResNet1D(input_channels=12, num_classes=5)
        optimizer = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "ckpt.pt")

        save_checkpoint(model, optimizer, epoch=1, metrics={}, path=path, is_best=True)

        assert (tmp_path / "best_model.pt").exists()
