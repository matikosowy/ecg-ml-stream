"""Unit tests for ECGDataset.

Copyright 2026 Mateusz Golebiewski
"""

from unittest.mock import patch

import pytest
import torch

from ecg.dataset.ecg_dataset import ECGDataset, create_dataloaders

RDSAMP = "ecg.dataset.ecg_dataset.wfdb.rdsamp"


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
        with patch(RDSAMP, return_value=(fake_signal_100hz, {})):
            signal, label = ds[0]
        assert isinstance(signal, torch.Tensor)
        assert isinstance(label, int)

    def test_500hz_getitem(self, csv_mocks, fake_signal_500hz, tmp_path):
        ds = ECGDataset(str(tmp_path), sampling_rate=500, split="train")
        with patch(RDSAMP, return_value=(fake_signal_500hz, {})):
            signal, _ = ds[0]
        assert isinstance(signal, torch.Tensor)

    def test_get_full_record(self, csv_mocks, fake_signal_100hz, tmp_path):
        ds = ECGDataset(str(tmp_path), sampling_rate=100, split="train")
        ecg_id = ds.records.index[0]
        with patch(RDSAMP, return_value=(fake_signal_100hz, {})):
            signal, label = ds.get_full_record(ecg_id)
        assert signal.shape == (12, 1000)
        assert isinstance(label, int)

    def test_get_record_windows(self, csv_mocks, fake_signal_100hz, tmp_path):
        ds = ECGDataset(str(tmp_path), sampling_rate=100, split="train")
        ecg_id = ds.records.index[0]
        with patch(RDSAMP, return_value=(fake_signal_100hz, {})):
            windows, label = ds.get_record_windows(ecg_id)
        assert windows.ndim == 3
        assert isinstance(label, int)


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
