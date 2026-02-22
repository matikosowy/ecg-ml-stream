"""PTB-XL dataset loader module for 12-lead ECG classification.

Copyright 2026 Mateusz Golebiewski
"""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset

from ecg_ml_stream.utils.constants import CLASS_NAMES, SUPERCLASS_MAPPING, SUPERCLASS_PRIORITY
from ecg_ml_stream.utils.helpers import normalize_signal


class ECGDataset(Dataset):
    """PyTorch Dataset for the PTB-XL ECG dataset.

    Attributes:
        CLASS_NAMES (ClassVar[list[str]]): List of class names.
        SUPERCLASS_MAPPING (ClassVar[dict[str, int]]): Mapping of superclass names to label indices.

    """

    CLASS_NAMES: ClassVar[list[str]] = CLASS_NAMES
    SUPERCLASS_MAPPING: ClassVar[dict[str, int]] = SUPERCLASS_MAPPING

    def __init__(
        self,
        data_path: str,
        sampling_rate: int = 100,
        window_size: float = 2.5,
        window_stride: float = 1.25,
        split: str = "train",
        transforms: Callable | None = None,
    ) -> None:
        """Initialize the ECGDataset.

        Args:
            data_path (str): Root directory of the PTB-XL dataset.
            sampling_rate (int): Sampling rate of the ECG signals. Defaults to 100 Hz.
            window_size (float): Duration of a training window in seconds.
                Defaults to 2.5 seconds.
            window_stride (float): Stride between training windows in seconds.
                Defaults to 1.25 seconds.
            split (str): Dataset split to use ('train', 'val', 'test'). Defaults to 'train'.
            transforms (callable): Optional. Transformations to apply to each window.

        """
        self.data_path = Path(data_path)
        self.sampling_rate = sampling_rate
        self.window_size = int(window_size * sampling_rate)
        self.window_stride = int(window_stride * sampling_rate)
        self.split = split
        self.transforms = transforms

        self.metadata = self._load_metadata()
        self.records = self._get_split_records()
        self.window_index = self._create_window_index()

    def _load_metadata(self) -> pd.DataFrame:
        """Load and preprocess the PTB-XL metadata.

        Returns:
            pd.DataFrame: Preprocessed metadata DataFrame.

        """
        metadata_path = self.data_path / "ptbxl_database.csv"
        df = pd.read_csv(metadata_path, index_col="ecg_id")
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)  # Parse strings to dicts

        scp_path = self.data_path / "scp_statements.csv"
        scp_df = pd.read_csv(scp_path, index_col=0)
        scp_df = scp_df[scp_df["diagnostic"] == 1]

        def get_superclass(scp_codes: dict) -> str | None:
            """Map record's SCP codes to a superclass of highest priority.

            Args:
                scp_codes (dict): Dictionary mapping SCP code to confidence.

            Returns:
                str: Superclass name.

            """
            superclasses = []
            for code, confidence in scp_codes.items():
                if code in scp_df.index and confidence > 0:
                    superclass = scp_df.loc[code, "diagnostic_class"]
                    if pd.notna(superclass) and superclass in self.SUPERCLASS_MAPPING:
                        superclasses.append(superclass)

            for superclass in SUPERCLASS_PRIORITY:
                if superclass in superclasses:
                    return superclass
            return None

        df["superclass"] = df["scp_codes"].apply(get_superclass)

        df = df[df["superclass"].notna()]
        df["label"] = df["superclass"].map(self.SUPERCLASS_MAPPING)

        return df

    def _get_split_records(self) -> pd.DataFrame:
        """Filter metadata to the requested train/val/test split.

        PTB-XL uses stratified folds 1-10: fold 9 - val, fold 10 - test, rest - train.

        Returns:
            pd.DataFrame: Metadata for the requested split.

        Raises:
            ValueError: If an invalid split name is provided.

        """
        if self.split == "train":
            return self.metadata[self.metadata["strat_fold"] <= 8]
        if self.split == "val":
            return self.metadata[self.metadata["strat_fold"] == 9]
        if self.split == "test":
            return self.metadata[self.metadata["strat_fold"] == 10]
        msg = f"Unknown split: {self.split}. Valid options are 'train', 'val' or 'test'."
        raise ValueError(msg)

    def _create_window_index(self) -> list[tuple[int, int, int]]:
        """Build an index mapping dataset records to (ecg_id, window_start, label) tuples.

        Returns:
            list[tuple[int, int, int]]: List of tuples containing (ecg_id, window_start, label).

        """
        windows: list[tuple[int, int, int]] = []
        signal_length = 10 * self.sampling_rate  # Each record is 10 seconds long

        for ecg_id, row in self.records.iterrows():
            start = 0
            while start + self.window_size <= signal_length:
                windows.append((ecg_id, start, row["label"]))
                start += self.window_stride

        return windows

    def _load_signal(self, ecg_id: int) -> np.ndarray:
        """Load a raw ECG signal from the dataset.

        Args:
            ecg_id (int): Unique identifier of the ECG record. Index in the metadata DataFrame.

        Returns:
            np.ndarray: Raw ECG signal of shape (12, signal_length).

        """
        row = self.metadata.loc[ecg_id]
        key = "filename_lr" if self.sampling_rate == 100 else "filename_hr"
        filename = row[key]
        filepath = self.data_path / filename
        signal, _ = wfdb.rdsamp(str(filepath))  # Returns (samples, channels)

        return signal.T.astype(np.float32)  # Transpose to (channels, samples)

    def __len__(self) -> int:
        """Return the total number of windows in the split.

        Returns:
            int: Number of split items.

        """
        return len(self.window_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a single normalized ECG window and its label.

        Args:
            idx (int): Index of the window to retrieve.

        Returns:
            tuple[torch.Tensor, int]: Tuple containing the ECG window
            tensor of shape (12, window_size) and its label.

        """
        ecg_id, window_start, label = self.window_index[idx]

        signal = self._load_signal(ecg_id)
        window = signal[:, window_start : window_start + self.window_size]
        window = normalize_signal(window)
        window_tensor = torch.from_numpy(window)

        if self.transforms:
            window_tensor = self.transforms(window_tensor)

        return window_tensor, label

    def get_full_record(self, ecg_id: int) -> tuple[np.ndarray, int]:
        """Return the full 10-second signal and label for a record.

        Args:
            ecg_id (int): Unique PTB-XL record indentifier.

        Returns:
            tuple[np.ndarray, int]: Tuple containing the full ECG signal
            of shape (12, samples) and its label.

        """
        signal = self._load_signal(ecg_id)
        label = int(self.metadata.loc[ecg_id, "label"])
        return signal, label

    def get_record_windows(self, ecg_id: int) -> tuple[torch.Tensor, int]:
        """Return all sliding windows for a record (for inference).

        Args:
        ecg_id (int): Unique PTB-XL record indentifier.

        Returns:
            tuple[torch.Tensor, int]: Tuple containing (windows_tensor, label_index)
            where windows has shape (num_windows, 12, window_size).

        """
        signal = self._load_signal(ecg_id)
        label = int(self.metadata.loc[ecg_id, "label"])

        windows = []
        signal_length = signal.shape[1]
        start = 0

        while start + self.window_size <= signal_length:
            window = normalize_signal(signal[:, start : start + self.window_size])
            windows.append(window)
            start += self.window_stride

        return torch.from_numpy(np.stack(windows, axis=0)), label

    def get_sample_for_streaming(self, idx: int | None = None) -> dict:
        """Return a record formatted for Kafka streaming.

        Args:
            idx (int | None): Optional index of the record to retrieve.
                If None, a random record is returned.

        Returns:
            dict: Dictionary containing the ECG record data and metadata,
                ready for JSON serialization.

        """
        if idx is None:
            idx = int(np.random.default_rng().integers(len(self.records)))

        ecg_id = self.records.index[idx]
        row = self.records.loc[ecg_id]
        signal = self._load_signal(ecg_id)

        return {
            "ecg_id": int(ecg_id),
            "signal": signal.tolist(),
            "label": int(row["label"]),
            "label_name": self.CLASS_NAMES[int(row["label"])],
            "patient_id": (int(row["patient_id"]) if pd.notna(row["patient_id"]) else None),
            "age": int(row["age"]) if pd.notna(row["age"]) else None,
            "sex": int(row["sex"]) if pd.notna(row["sex"]) else None,
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for the training split.

        Returns:
            torch.Tensor: Tensor of shape (num_classes,) with per-class weights.

        """
        labels = [w[2] for w in self.window_index]
        class_counts = np.bincount(labels, minlength=5)
        total = len(labels)
        weights = total / (5 * class_counts + 1e-8)
        return torch.FloatTensor(weights)


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    sampling_rate: int = 100,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for the train, validation and test splits.

    Args:
        data_path (str): Root directory of the PTB-XL dataset.
        batch_size (int): Batch size for the DataLoaders. Defaults to 32.
        sampling_rate (int): Sampling rate of the ECG signals. Defaults to 100 Hz.
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Tuple containing the train, validation
        and test DataLoaders.

    """
    train_dataset = ECGDataset(data_path, sampling_rate, split="train")
    val_dataset = ECGDataset(data_path, sampling_rate, split="val")
    test_dataset = ECGDataset(data_path, sampling_rate, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class ECGAugmentation:
    """On-the-fly ECG signal augmentation applied during training."""

    def __init__(
        self,
        amplitude_scale_range: tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.05,
        lead_dropout_prob: float = 0.1,
        time_mask_max_samples: int = 25,
        p: float = 0.5,
    ) -> None:
        """Initialize ECGAugmentation.

        Args:
            amplitude_scale_range (tuple[float, float]): Range for random amplitude scaling.
                Defaults to (0.8, 1.2).
            noise_std (float): Standard deviation of the added Gaussian noise. Defaults to 0.05.
            lead_dropout_prob (float): Probability of dropping out each lead. Defaults to 0.1.
            time_mask_max_samples (int): Maximum length of time masking in samples. Defaults to 25.
            p (float): Probability of applying augmentation to a sample. Defaults to 0.5.

        """
        self.amplitude_scale_range = amplitude_scale_range
        self.noise_std = noise_std
        self.lead_dropout_prob = lead_dropout_prob
        self.time_mask_max_samples = time_mask_max_samples
        self.p = p
        self._rng = np.random.default_rng()

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a signal tensor.

        Args:
            signal (torch.Tensor): Input ECG signal tensor of shape (12, samples).

        Returns:
            torch.Tensor: Augmented ECG signal tensor of shape (12, samples).

        """
        x = signal.clone()

        # Amplitude scaling
        if self._rng.random() < self.p:
            scale = self._rng.uniform(*self.amplitude_scale_range)
            x *= scale

        # Additive Gaussian noise
        if self._rng.random() < self.p:
            noise = torch.randn_like(x) * self.noise_std
            x += noise

        # Lead dropout
        if self._rng.random() < self.p:
            mask = torch.from_numpy(
                self._rng.random(x.shape[0]) > self.lead_dropout_prob,
            ).float()
            x *= mask.unsqueeze(1)

        # Time masking
        if self._rng.random() < self.p:
            seq_len = x.shape[1]
            mask_len = self._rng.integers(1, self.time_mask_max_samples + 1)
            start = self._rng.integers(0, max(1, seq_len - mask_len))
            x[:, start : start + mask_len] = 0.0

        return x
