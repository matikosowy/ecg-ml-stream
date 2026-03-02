"""Unit tests for inference module.

Copyright 2026 Mateusz Golebiewski
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ecg_ml_stream.ml.inference import _cache, get_model, infer_ecg_record

_INFERENCE = "ecg_ml_stream.ml.inference"
_ECG_CLASSIFIER = f"{_INFERENCE}.ECGClassifier"


def _fake_diagnosis() -> dict:
    return {
        "class": "NORM",
        "class_idx": 0,
        "probability": 0.9,
        "all_probabilities": {"NORM": 0.9, "MI": 0.025, "STTC": 0.025, "CD": 0.025, "HYP": 0.025},
        "is_dangerous": False,
        "description": "Normal sinus rhythm",
    }


def _make_signal(n_samples: int = 1000) -> list[list[float]]:
    return np.random.default_rng(0).standard_normal((12, n_samples)).tolist()


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    _cache.instance = None
    _cache.path = None
    yield
    _cache.instance = None
    _cache.path = None


class TestGetModel:
    def test_loads_from_existing_path(self, tmp_path):
        model_file = tmp_path / "model.pt"
        model_file.touch()
        mock_instance = MagicMock()

        with patch(_ECG_CLASSIFIER, return_value=mock_instance) as mock_cls:
            result = get_model(str(model_file))

        mock_cls.assert_called_once_with(model_path=str(model_file))
        assert result is mock_instance

    def test_uses_random_weights_when_path_missing(self):
        mock_instance = MagicMock()

        with patch(_ECG_CLASSIFIER, return_value=mock_instance) as mock_cls:
            result = get_model("/nonexistent/model.pt")

        mock_cls.assert_called_once_with()
        assert result is mock_instance

    def test_caches_instance_on_same_path(self, tmp_path):
        model_file = tmp_path / "model.pt"
        model_file.touch()
        mock_instance = MagicMock()

        with patch(_ECG_CLASSIFIER, return_value=mock_instance) as mock_cls:
            first = get_model(str(model_file))
            second = get_model(str(model_file))

        assert mock_cls.call_count == 1
        assert first is second

    def test_reloads_on_path_change(self, tmp_path):
        path1 = str(tmp_path / "model1.pt")
        path2 = str(tmp_path / "model2.pt")
        (tmp_path / "model1.pt").touch()
        (tmp_path / "model2.pt").touch()

        instance1, instance2 = MagicMock(), MagicMock()

        with patch(_ECG_CLASSIFIER, side_effect=[instance1, instance2]) as mock_cls:
            first = get_model(path1)
            second = get_model(path2)

        assert mock_cls.call_count == 2
        assert first is instance1
        assert second is instance2

    def test_cache_path_updated(self, tmp_path):
        model_file = tmp_path / "model.pt"
        model_file.touch()

        with patch(_ECG_CLASSIFIER, return_value=MagicMock()):
            get_model(str(model_file))

        assert _cache.path == str(model_file)

    def test_returns_cached_instance_without_reload(self):
        mock_instance = MagicMock()
        _cache.instance = mock_instance
        _cache.path = "/some/path.pt"

        with patch(_ECG_CLASSIFIER) as mock_cls:
            result = get_model("/some/path.pt")

        mock_cls.assert_not_called()
        assert result is mock_instance


class TestInferEcgRecord:
    def test_100hz_correct_window_size(self):
        mock_classifier = MagicMock()
        mock_classifier.predict_windows.return_value = _fake_diagnosis()

        with patch(f"{_INFERENCE}.get_model", return_value=mock_classifier):
            infer_ecg_record(_make_signal(1000), sampling_rate=100)

        tensor_arg = mock_classifier.predict_windows.call_args[0][0]
        assert isinstance(tensor_arg, torch.Tensor)
        assert tensor_arg.shape[1] == 12
        assert tensor_arg.shape[2] == 250

    def test_100hz_correct_stride_produces_windows(self):
        mock_classifier = MagicMock()
        mock_classifier.predict_windows.return_value = _fake_diagnosis()

        with patch(f"{_INFERENCE}.get_model", return_value=mock_classifier):
            infer_ecg_record(_make_signal(1000), sampling_rate=100)

        tensor_arg = mock_classifier.predict_windows.call_args[0][0]
        # 1000 samples, window=250, stride=125 → 7 windows
        assert tensor_arg.shape[0] == 7

    def test_500hz_correct_window_size(self):
        mock_classifier = MagicMock()
        mock_classifier.predict_windows.return_value = _fake_diagnosis()

        with patch(f"{_INFERENCE}.get_model", return_value=mock_classifier):
            infer_ecg_record(_make_signal(5000), sampling_rate=500)

        tensor_arg = mock_classifier.predict_windows.call_args[0][0]
        assert tensor_arg.shape[2] == 1250

    def test_500hz_correct_window_count(self):
        mock_classifier = MagicMock()
        mock_classifier.predict_windows.return_value = _fake_diagnosis()

        with patch(f"{_INFERENCE}.get_model", return_value=mock_classifier):
            infer_ecg_record(_make_signal(5000), sampling_rate=500)

        tensor_arg = mock_classifier.predict_windows.call_args[0][0]
        # 5000 samples, window=1250, stride=625 → 7 windows
        assert tensor_arg.shape[0] == 7

    def test_returns_classifier_result(self):
        expected = _fake_diagnosis()
        mock_classifier = MagicMock()
        mock_classifier.predict_windows.return_value = expected

        with patch(f"{_INFERENCE}.get_model", return_value=mock_classifier):
            result = infer_ecg_record(_make_signal(1000), sampling_rate=100)

        assert result is expected

    def test_tensor_dtype_is_float32(self):
        mock_classifier = MagicMock()
        mock_classifier.predict_windows.return_value = _fake_diagnosis()

        with patch(f"{_INFERENCE}.get_model", return_value=mock_classifier):
            infer_ecg_record(_make_signal(1000), sampling_rate=100)

        tensor_arg = mock_classifier.predict_windows.call_args[0][0]
        assert tensor_arg.dtype == torch.float32

    def test_custom_model_path_forwarded(self, tmp_path):
        mock_classifier = MagicMock()
        mock_classifier.predict_windows.return_value = _fake_diagnosis()
        custom_path = "/custom/model.pt"

        with patch(f"{_INFERENCE}.get_model", return_value=mock_classifier) as mock_get:
            infer_ecg_record(_make_signal(1000), sampling_rate=100, model_path=custom_path)

        mock_get.assert_called_once_with(custom_path)
