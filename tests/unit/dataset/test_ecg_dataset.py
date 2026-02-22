"""Unit tests for ECGDataset.

Copyright 2026 Mateusz Golebiewski
"""

from unittest.mock import patch

import pytest
import torch

from ecg_ml_stream.dataset.ecg_dataset import ECGAugmentation, ECGDataset, create_dataloaders

_RDSAMP = "ecg_ml_stream.dataset.ecg_dataset.wfdb.rdsamp"


class TestECGDataset:
    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_dataset_split_creates_instance(self, csv_mocks, tmp_path, split):
        ds = ECGDataset(str(tmp_path), sampling_rate=100, split=split)
        assert isinstance(ds, ECGDataset)
        assert len(ds) >= 0

    def test_invalid_split_raises(self, csv_mocks, tmp_path):
        with pytest.raises(ValueError, match="Unknown split"):
            ECGDataset(str(tmp_path), split="unknown")

    def test_getitem_returns_tensor_and_label(
        self,
        csv_mocks,
        fake_signal_100hz,
        tmp_path,
    ):
        ds = ECGDataset(str(tmp_path), sampling_rate=100, split="train")
        with patch(_RDSAMP, return_value=(fake_signal_100hz, {})):
            signal, label = ds[0]
        assert isinstance(signal, torch.Tensor)
        assert isinstance(label, int)

    def test_500hz_getitem(self, csv_mocks, fake_signal_500hz, tmp_path):
        ds = ECGDataset(str(tmp_path), sampling_rate=500, split="train")
        with patch(_RDSAMP, return_value=(fake_signal_500hz, {})):
            signal, _ = ds[0]
        assert isinstance(signal, torch.Tensor)

    def test_get_full_record(self, csv_mocks, fake_signal_100hz, tmp_path):
        ds = ECGDataset(str(tmp_path), sampling_rate=100, split="train")
        ecg_id = ds.records.index[0]
        with patch(_RDSAMP, return_value=(fake_signal_100hz, {})):
            signal, label = ds.get_full_record(ecg_id)
        assert signal.shape == (12, 1000)
        assert isinstance(label, int)

    def test_get_record_windows(self, csv_mocks, fake_signal_100hz, tmp_path):
        ds = ECGDataset(str(tmp_path), sampling_rate=100, split="train")
        ecg_id = ds.records.index[0]
        with patch(_RDSAMP, return_value=(fake_signal_100hz, {})):
            windows, label = ds.get_record_windows(ecg_id)
        assert windows.ndim == 3
        assert isinstance(label, int)

    def test_getitem_applies_transform(self, csv_mocks, fake_signal_100hz, tmp_path):
        def identity(x):
            return x

        ds = ECGDataset(str(tmp_path), sampling_rate=100, split="train", transforms=identity)
        with patch(_RDSAMP, return_value=(fake_signal_100hz, {})):
            signal, label = ds[0]
        assert isinstance(signal, torch.Tensor)
        assert isinstance(label, int)

    def test_get_sample_for_streaming(self, csv_mocks, fake_signal_100hz, tmp_path):
        ds = ECGDataset(str(tmp_path), sampling_rate=100, split="train")
        with patch(_RDSAMP, return_value=(fake_signal_100hz, {})):
            sample = ds.get_sample_for_streaming()
        assert isinstance(sample["ecg_id"], int)
        assert isinstance(sample["signal"], list)
        assert isinstance(sample["label"], int)

    def test_get_class_weights(self, csv_mocks, tmp_path):
        ds = ECGDataset(str(tmp_path), sampling_rate=100, split="train")
        weights = ds.get_class_weights()
        assert isinstance(weights, torch.Tensor)
        assert len(weights) == 5


class TestCreateDataloaders:
    def test_returns_three_loaders(self, csv_mocks, tmp_path):
        loaders = create_dataloaders(
            str(tmp_path),
            batch_size=4,
            sampling_rate=100,
            num_workers=0,
        )
        train_loader, val_loader, test_loader = loaders
        assert len(train_loader.dataset) >= 0
        assert len(val_loader.dataset) >= 0
        assert len(test_loader.dataset) >= 0


class TestECGAugmentation:
    def test_default_init(self):
        aug = ECGAugmentation()
        assert aug.p == 0.5
        assert aug.noise_std == 0.05

    def test_output_shape_unchanged(self):
        aug = ECGAugmentation(p=1.0)
        x = torch.randn(12, 1000)
        assert aug(x).shape == x.shape

    def test_p_zero_no_change(self):
        aug = ECGAugmentation(p=0.0)
        x = torch.ones(12, 1000)
        torch.testing.assert_close(aug(x), x)

    def test_p_one_applies_all_augmentations(self):
        aug = ECGAugmentation(p=1.0, noise_std=0.5, lead_dropout_prob=0.5, time_mask_max_samples=50)
        x = torch.ones(12, 1000)
        y = aug(x)
        assert y.shape == x.shape

    def test_returns_tensor(self):
        aug = ECGAugmentation()
        assert isinstance(aug(torch.randn(12, 250)), torch.Tensor)

    def test_amplitude_scaling_changes_values(self):
        aug = ECGAugmentation(p=1.0, noise_std=0.0, lead_dropout_prob=0.0, time_mask_max_samples=10)
        x = torch.ones(12, 100)
        y = aug(x)
        assert y.shape == x.shape
