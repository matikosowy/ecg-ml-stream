"""Unit tests for ResNet1D model.

Copyright 2026 Mateusz Golebiewski
"""

import pytest
import torch

from ecg_ml_stream.ml.model import ECGClassifier, ResNet1D, create_model
from ecg_ml_stream.utils.constants import CLASS_NAMES, DANGEROUS_CLASSES


class TestResNet1D:
    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_output_shape(self, untrained_model, batch_size):
        x = torch.randn(batch_size, 12, 250)
        out = untrained_model(x)
        assert out.shape == (batch_size, 5)

    def test_predict_proba_sums_to_one(self, untrained_model):
        x = torch.randn(4, 12, 250)
        probs = untrained_model.predict_proba(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_parameter_count(self):
        model = ResNet1D(input_channels=12, num_classes=5)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        assert param_count < 50_000_000

    def test_default_num_blocks(self):
        model = ResNet1D()
        x = torch.randn(2, 12, 250)
        assert model(x).shape == (2, 5)

    def test_predict_returns_class_index(self, untrained_model):
        x = torch.randn(4, 12, 250)
        preds = untrained_model.predict(x)
        assert preds.shape == (4,)
        assert preds.min() >= 0
        assert preds.max() < 5


class TestECGClassifier:
    def test_predict_windows_keys(self, untrained_classifier):
        windows = torch.randn(7, 12, 250)
        result = untrained_classifier.predict_windows(windows)

        expected_keys = {
            "class",
            "class_idx",
            "probability",
            "all_probabilities",
            "is_dangerous",
            "description",
            "window_predictions",
        }
        assert expected_keys == set(result.keys())

    def test_predict_windows_valid_class(self, untrained_classifier):
        windows = torch.randn(7, 12, 250)
        result = untrained_classifier.predict_windows(windows)
        assert result["class"] in CLASS_NAMES

    def test_predict_windows_probability_range(self, untrained_classifier):
        windows = torch.randn(7, 12, 250)
        result = untrained_classifier.predict_windows(windows)
        assert 0.0 <= result["probability"] <= 1.0

        for prob in result["all_probabilities"].values():
            assert 0.0 <= prob <= 1.0

    def test_predict_windows_is_dangerous(self, untrained_classifier):
        windows = torch.randn(7, 12, 250)
        result = untrained_classifier.predict_windows(windows)

        if result["class"] in DANGEROUS_CLASSES:
            assert result["is_dangerous"] is True
        else:
            assert result["is_dangerous"] is False

    def test_predict_single(self, untrained_classifier):
        signal = torch.randn(12, 250)
        result = untrained_classifier.predict_single(signal)
        assert result["class"] in CLASS_NAMES
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_single_with_batch_dim(self, untrained_classifier):
        signal = torch.randn(1, 12, 250)
        result = untrained_classifier.predict_single(signal)
        assert result["class"] in CLASS_NAMES

    def test_window_predictions_shape(self, untrained_classifier):
        windows = torch.randn(7, 12, 250)
        result = untrained_classifier.predict_windows(windows)
        assert len(result["window_predictions"]) == 7
        assert len(result["window_predictions"][0]) == 5

    def test_predict_windows_raises_for_wrong_channels(self, untrained_classifier):
        wrong = torch.randn(7, 6, 250)  # 6 channels instead of 12
        with pytest.raises(ValueError, match="Expected tensor of shape"):
            untrained_classifier.predict_windows(wrong)

    def test_predict_windows_raises_for_2d_tensor(self, untrained_classifier):
        wrong = torch.randn(12, 250)  # 2D instead of 3D
        with pytest.raises(ValueError, match="Expected tensor of shape"):
            untrained_classifier.predict_windows(wrong)


class TestECGClassifierSaveLoad:
    def test_save_creates_file(self, tmp_path, untrained_classifier):
        path = str(tmp_path / "model.pt")
        untrained_classifier.save(path)
        assert (tmp_path / "model.pt").exists()

    def test_save_with_optimizer_epoch_metrics(self, tmp_path, untrained_classifier):
        optimizer = torch.optim.Adam(untrained_classifier.model.parameters())
        path = str(tmp_path / "full.pt")
        untrained_classifier.save(path, optimizer=optimizer, epoch=5, metrics={"f1": 0.9})

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 5
        assert checkpoint["metrics"]["f1"] == pytest.approx(0.9)
        assert "optimizer_state_dict" in checkpoint

    def test_load_from_saved_checkpoint(self, tmp_path, untrained_classifier):
        path = str(tmp_path / "ckpt.pt")
        untrained_classifier.save(path)
        untrained_classifier.load(path)  # state_dict only format

    def test_load_from_full_checkpoint(self, tmp_path, untrained_classifier):
        path = str(tmp_path / "full.pt")
        untrained_classifier.save(path, epoch=2, metrics={})
        untrained_classifier.load(path)  # checkpoint with model_state_dict key

    def test_load_raw_state_dict(self, tmp_path, untrained_classifier):
        # Save raw state dict (no wrapping dict)
        path = str(tmp_path / "raw.pt")
        torch.save(untrained_classifier.model.state_dict(), path)
        untrained_classifier.load(path)

    def test_init_with_model_path(self, tmp_path, untrained_classifier):
        path = str(tmp_path / "model.pt")
        untrained_classifier.save(path)

        clf2 = ECGClassifier(model_path=path)
        windows = torch.randn(3, 12, 250)
        result = clf2.predict_windows(windows)
        assert result["class"] in clf2.CLASS_NAMES


class TestCreateModel:
    def test_default_params(self):
        model = create_model()
        assert isinstance(model, ResNet1D)
        x = torch.randn(2, 12, 250)
        assert model(x).shape == (2, 5)

    def test_custom_classes(self):
        model = create_model(input_channels=12, num_classes=3)
        x = torch.randn(2, 12, 250)
        assert model(x).shape == (2, 3)

    def test_with_pretrained_full_checkpoint(self, tmp_path, untrained_classifier):
        # create_model loading a full checkpoint (has model_state_dict key)
        path = str(tmp_path / "pretrained.pt")
        untrained_classifier.save(path)
        model = create_model(pretrained_path=path)
        assert isinstance(model, ResNet1D)

    def test_with_pretrained_raw_state_dict(self, tmp_path):
        # create_model loading a raw state dict
        base = create_model()
        path = str(tmp_path / "raw.pt")
        torch.save(base.state_dict(), path)
        model = create_model(pretrained_path=path)
        assert isinstance(model, ResNet1D)
