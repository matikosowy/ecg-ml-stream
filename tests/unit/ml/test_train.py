"""Unit tests for training module.

Copyright 2026 Mateusz Golebiewski
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_ml_stream.ml.model import ResNet1D
from ecg_ml_stream.ml.train import (
    evaluate_with_voting,
    main,
    parse_args,
    train_epoch,
    validate,
    voting_prediction,
)
from ecg_ml_stream.utils.helpers import save_checkpoint

_ECG_CLASSIFIER = "ecg_ml_stream.ml.train.ECGClassifier"
_TRAIN = "ecg_ml_stream.ml.train"


def _fake_metrics(f1_macro: float = 0.4) -> dict:
    return {"loss": 0.5, "accuracy": 0.7, "f1_macro": f1_macro}


def _make_fake_ds(n_samples: int = 20, seq_len: int = 250) -> TensorDataset:
    ds = TensorDataset(
        torch.randn(n_samples, 12, seq_len),
        torch.randint(0, 5, (n_samples,)),
    )
    ds.get_class_weights = lambda: torch.ones(5)
    ds.records = MagicMock()
    ds.records.index.tolist.return_value = list(range(n_samples))
    return ds


def _make_loader(n_samples: int = 16, batch_size: int = 4, seq_len: int = 250) -> DataLoader:
    signals = torch.randn(n_samples, 12, seq_len)
    labels = torch.randint(0, 5, (n_samples,))
    return DataLoader(TensorDataset(signals, labels), batch_size=batch_size)


class TestParseArgs:
    def test_default_lr(self):
        with patch("sys.argv", ["train"]):
            args = parse_args()
        assert args.lr == pytest.approx(1e-3)

    def test_default_epochs(self):
        with patch("sys.argv", ["train"]):
            args = parse_args()
        assert args.epochs == 50

    def test_default_batch_size(self):
        with patch("sys.argv", ["train"]):
            args = parse_args()
        assert args.batch_size == 64

    def test_default_seed(self):
        with patch("sys.argv", ["train"]):
            args = parse_args()
        assert args.seed == 42

    def test_resume_defaults_to_none(self):
        with patch("sys.argv", ["train"]):
            args = parse_args()
        assert args.resume is None

    def test_custom_lr(self):
        with patch("sys.argv", ["train", "--lr", "0.01"]):
            args = parse_args()
        assert args.lr == pytest.approx(0.01)

    def test_custom_epochs(self):
        with patch("sys.argv", ["train", "--epochs", "10"]):
            args = parse_args()
        assert args.epochs == 10

    def test_sampling_rate_500(self):
        with patch("sys.argv", ["train", "--sampling-rate", "500"]):
            args = parse_args()
        assert args.sampling_rate == 500

    def test_resume_path(self):
        with patch("sys.argv", ["train", "--resume", "/path/ckpt.pt"]):
            args = parse_args()
        assert args.resume == "/path/ckpt.pt"


class TestTrainEpoch:
    def test_returns_expected_keys(self, untrained_model):
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(untrained_model.parameters(), lr=1e-3)

        result = train_epoch(
            untrained_model, loader, criterion, optimizer, torch.device("cpu"), epoch=1
        )

        assert "loss" in result
        assert "accuracy" in result
        assert "f1_macro" in result

    def test_loss_is_positive(self, untrained_model):
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(untrained_model.parameters(), lr=1e-3)

        result = train_epoch(
            untrained_model, loader, criterion, optimizer, torch.device("cpu"), epoch=1
        )

        assert result["loss"] > 0

    def test_accuracy_in_range(self, untrained_model):
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(untrained_model.parameters(), lr=1e-3)

        result = train_epoch(
            untrained_model, loader, criterion, optimizer, torch.device("cpu"), epoch=1
        )

        assert 0.0 <= result["accuracy"] <= 1.0

    def test_weights_change_after_training(self, untrained_model):
        initial = untrained_model.fc.weight.data.clone()
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(untrained_model.parameters(), lr=1e-3)

        train_epoch(untrained_model, loader, criterion, optimizer, torch.device("cpu"), epoch=1)

        assert not torch.allclose(initial, untrained_model.fc.weight.data)


class TestValidate:
    def test_returns_expected_keys(self, untrained_model):
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()

        result = validate(untrained_model, loader, criterion, torch.device("cpu"), epoch=1)

        assert "loss" in result
        assert "accuracy" in result
        assert "f1_macro" in result

    def test_model_in_eval_mode_after_call(self, untrained_model):
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()

        validate(untrained_model, loader, criterion, torch.device("cpu"), epoch=1)

        assert not untrained_model.training

    def test_weights_unchanged(self, untrained_model):
        initial = untrained_model.fc.weight.data.clone()
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()

        validate(untrained_model, loader, criterion, torch.device("cpu"), epoch=1)

        assert torch.allclose(initial, untrained_model.fc.weight.data)

    def test_loss_is_positive(self, untrained_model):
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()

        result = validate(untrained_model, loader, criterion, torch.device("cpu"), epoch=1)

        assert result["loss"] > 0


class TestEvaluateWithVoting:
    def _make_mock_dataset(self, record_ids, label=0):
        mock = MagicMock()
        mock.records.index.tolist.return_value = record_ids
        mock.get_record_windows.return_value = (torch.randn(7, 12, 250), label)
        return mock

    def test_returns_expected_keys(self, untrained_model):
        with patch(_ECG_CLASSIFIER) as mock_classifier:
            mock_classifier.return_value.predict_windows.return_value = {"class_idx": 0}

            result = evaluate_with_voting(
                untrained_model, self._make_mock_dataset([1, 2, 3]), torch.device("cpu")
            )

        assert "voting_accuracy" in result
        assert "voting_f1_macro" in result
        assert "total_records" in result

    def test_total_records_equals_dataset_size(self, untrained_model):
        with patch(_ECG_CLASSIFIER) as mock_classifier:
            mock_classifier.return_value.predict_windows.return_value = {"class_idx": 0}

            result = evaluate_with_voting(
                untrained_model, self._make_mock_dataset(list(range(8))), torch.device("cpu")
            )

        assert result["total_records"] == 8

    def test_num_samples_limits_records(self, untrained_model):
        with patch(_ECG_CLASSIFIER) as mock_classifier:
            mock_classifier.return_value.predict_windows.return_value = {"class_idx": 0}

            result = evaluate_with_voting(
                untrained_model,
                self._make_mock_dataset(list(range(10))),
                torch.device("cpu"),
                num_samples=3,
            )

        assert result["total_records"] == 3

    def test_perfect_predictions(self, untrained_model):
        labels = [0, 1, 2, 3, 4]
        with patch(_ECG_CLASSIFIER) as mock_classifier:
            mock_classifier.return_value.predict_windows.side_effect = [
                {"class_idx": lbl} for lbl in labels
            ]
            mock_dataset = MagicMock()
            mock_dataset.records.index.tolist.return_value = list(range(5))
            mock_dataset.get_record_windows.side_effect = [
                (torch.randn(7, 12, 250), lbl) for lbl in labels
            ]

            result = evaluate_with_voting(untrained_model, mock_dataset, torch.device("cpu"))

        assert result["voting_accuracy"] == pytest.approx(1.0)
        assert result["voting_f1_macro"] == pytest.approx(1.0)

    def test_accuracy_range(self, untrained_model):
        with patch(_ECG_CLASSIFIER) as mock_classifier:
            mock_classifier.return_value.predict_windows.side_effect = [
                {"class_idx": 0},
                {"class_idx": 1},
                {"class_idx": 0},
            ]
            mock_dataset = MagicMock()
            mock_dataset.records.index.tolist.return_value = [1, 2, 3]
            mock_dataset.get_record_windows.side_effect = [
                (torch.randn(7, 12, 250), 0),
                (torch.randn(7, 12, 250), 0),
                (torch.randn(7, 12, 250), 1),
            ]

            result = evaluate_with_voting(untrained_model, mock_dataset, torch.device("cpu"))

        assert 0.0 <= result["voting_accuracy"] <= 1.0


class TestVotingPrediction:
    def test_soft_selects_highest_avg_class(self):
        preds = np.array([[0.1, 0.7, 0.2], [0.2, 0.6, 0.2], [0.3, 0.5, 0.2]], dtype=np.float32)
        class_idx, _ = voting_prediction(preds, mode="soft")
        assert class_idx == 1

    def test_soft_returns_correct_avg_probs(self):
        preds = np.array([[0.2, 0.5, 0.3], [0.4, 0.3, 0.3]], dtype=np.float32)
        _, avg_probs = voting_prediction(preds, mode="soft")
        np.testing.assert_allclose(avg_probs, preds.mean(axis=0), atol=1e-6)

    def test_hard_majority_vote(self):
        preds = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
        class_idx, _ = voting_prediction(preds, mode="hard")
        assert class_idx == 1

    def test_hard_returns_avg_probs(self):
        preds = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
        _, avg_probs = voting_prediction(preds, mode="hard")
        np.testing.assert_allclose(avg_probs, preds.mean(axis=0), atol=1e-6)

    def test_single_window_soft(self):
        preds = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
        class_idx, _ = voting_prediction(preds)
        assert class_idx == 2

    def test_returns_tuple_of_two(self):
        preds = np.array([[0.33, 0.33, 0.34]], dtype=np.float32)
        result = voting_prediction(preds)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_default_mode_is_soft(self):
        preds = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
        class_idx, avg_probs = voting_prediction(preds)
        assert class_idx == 1
        np.testing.assert_allclose(avg_probs, preds.mean(axis=0), atol=1e-6)


class TestMain:
    def test_exits_early_when_data_path_missing(self, tmp_path):
        argv = [
            "train",
            "--data-path",
            str(tmp_path / "nonexistent"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-workers",
            "0",
        ]
        with (
            patch("sys.argv", argv),
            patch(f"{_TRAIN}.get_device", return_value=torch.device("cpu")),
            patch(f"{_TRAIN}.setup_logging", return_value=MagicMock()),
            patch(f"{_TRAIN}.ECGDataset") as mock_ds_cls,
        ):
            main()

        mock_ds_cls.assert_not_called()

    def test_full_run_best_model_saved(self, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        fake_ds = _make_fake_ds()
        argv = [
            "train",
            "--data-path",
            str(data_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--epochs",
            "5",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--patience",
            "6",
        ]
        with (
            patch("sys.argv", argv),
            patch(f"{_TRAIN}.get_device", return_value=torch.device("cpu")),
            patch(f"{_TRAIN}.setup_logging", return_value=MagicMock()),
            patch(f"{_TRAIN}.ECGAugmentation"),
            patch(f"{_TRAIN}.ECGDataset", return_value=fake_ds),
            patch(f"{_TRAIN}.train_epoch", return_value=_fake_metrics(0.4)),
            patch(f"{_TRAIN}.validate", return_value=_fake_metrics(0.4)),
            patch(
                f"{_TRAIN}.evaluate_with_voting",
                return_value={
                    "voting_accuracy": 0.8,
                    "voting_f1_macro": 0.75,
                    "total_records": 20,
                },
            ),
        ):
            main()

        run_dirs = list((tmp_path / "out").glob("run_*"))
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "best_model.pt").exists()

    def test_full_run_no_best_model(self, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        fake_ds = _make_fake_ds()
        argv = [
            "train",
            "--data-path",
            str(data_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
        ]
        with (
            patch("sys.argv", argv),
            patch(f"{_TRAIN}.get_device", return_value=torch.device("cpu")),
            patch(f"{_TRAIN}.setup_logging", return_value=MagicMock()),
            patch(f"{_TRAIN}.ECGAugmentation"),
            patch(f"{_TRAIN}.ECGDataset", return_value=fake_ds),
            patch(f"{_TRAIN}.train_epoch", return_value=_fake_metrics(0.0)),
            patch(f"{_TRAIN}.validate", return_value=_fake_metrics(0.0)),
            patch(
                f"{_TRAIN}.evaluate_with_voting",
                return_value={
                    "voting_accuracy": 0.5,
                    "voting_f1_macro": 0.4,
                    "total_records": 20,
                },
            ),
        ):
            main()

        out_dir = tmp_path / "out"
        run_dirs = list(out_dir.glob("run_*"))
        assert len(run_dirs) == 1
        assert not (run_dirs[0] / "best_model.pt").exists()
        assert (out_dir / "ecg_resnet1d.pt").exists()

    def test_early_stopping_triggers(self, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        fake_ds = _make_fake_ds()
        argv = [
            "train",
            "--data-path",
            str(data_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--epochs",
            "10",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--patience",
            "2",
        ]
        with (
            patch("sys.argv", argv),
            patch(f"{_TRAIN}.get_device", return_value=torch.device("cpu")),
            patch(f"{_TRAIN}.setup_logging", return_value=MagicMock()),
            patch(f"{_TRAIN}.ECGAugmentation"),
            patch(f"{_TRAIN}.ECGDataset", return_value=fake_ds),
            patch(f"{_TRAIN}.train_epoch", return_value=_fake_metrics(0.4)),
            patch(f"{_TRAIN}.validate", return_value=_fake_metrics(0.4)),
            patch(
                f"{_TRAIN}.evaluate_with_voting",
                return_value={
                    "voting_accuracy": 0.5,
                    "voting_f1_macro": 0.4,
                    "total_records": 20,
                },
            ),
        ):
            main()

        run_dirs = list((tmp_path / "out").glob("run_*"))
        assert len(run_dirs) == 1
        history_file = run_dirs[0] / "history.json"
        assert history_file.exists()

    def test_resumes_from_checkpoint(self, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        ckpt_path = tmp_path / "ckpt.pth"

        resume_model = ResNet1D(input_channels=12, num_classes=5)
        resume_opt = torch.optim.AdamW(resume_model.parameters())
        save_checkpoint(
            resume_model,
            resume_opt,
            epoch=2,
            metrics={"val_f1_macro": 0.5},
            path=str(ckpt_path),
        )

        fake_ds = _make_fake_ds()
        argv = [
            "train",
            "--data-path",
            str(data_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--epochs",
            "3",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--resume",
            str(ckpt_path),
        ]
        with (
            patch("sys.argv", argv),
            patch(f"{_TRAIN}.get_device", return_value=torch.device("cpu")),
            patch(f"{_TRAIN}.setup_logging", return_value=MagicMock()),
            patch(f"{_TRAIN}.ECGAugmentation"),
            patch(f"{_TRAIN}.ECGDataset", return_value=fake_ds),
            patch(f"{_TRAIN}.train_epoch", return_value=_fake_metrics(0.0)),
            patch(f"{_TRAIN}.validate", return_value=_fake_metrics(0.0)),
            patch(
                f"{_TRAIN}.evaluate_with_voting",
                return_value={
                    "voting_accuracy": 0.5,
                    "voting_f1_macro": 0.4,
                    "total_records": 20,
                },
            ),
        ):
            main()

        assert (tmp_path / "out").exists()
