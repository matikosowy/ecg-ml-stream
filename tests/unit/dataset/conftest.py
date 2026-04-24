"""Fixtures for dataset unit tests in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _make_metadata(total: int = 12) -> pd.DataFrame:
    """Return a fake PTB-XL metadata DataFrame."""
    scp_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    base_codes = [f"{{'{scp_classes[i % 5]}': 100.0}}" for i in range(total - 2)]  # valid codes
    edge_codes = [
        "{'UNKNOWN_CODE': 100.0}",  # not in scp_df
        "{'ALIEN_CODE': 100.0}",  # in scp_df with unmapped superclass
    ]
    scp_codes = base_codes + edge_codes

    # Folds 1-8 train, 9 val, 10 test; last two are edge cases on fold 1
    folds = [1] * (total - 4) + [9, 10, 1, 1]
    return pd.DataFrame(
        {
            "scp_codes": scp_codes,
            "strat_fold": folds,
            "filename_lr": [f"records100/{i:05d}_lr" for i in range(1, total + 1)],
            "filename_hr": [f"records500/{i:05d}_hr" for i in range(1, total + 1)],
            "age": [50 + i for i in range(total)],
            "sex": [i % 2 for i in range(total)],
            "patient_id": range(1001, 1001 + total),
        },
        index=pd.Index(range(1, total + 1), name="ecg_id"),
    )


def _make_scp_df() -> pd.DataFrame:
    """Return a minimal fake SCP statements DataFrame."""
    return pd.DataFrame(
        {
            "diagnostic": [1, 1, 1, 1, 1, 1],
            "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", "ALIEN_CLASS"],
        },
        index=["NORM", "MI", "STTC", "CD", "HYP", "ALIEN_CODE"],
    )


def _csv_side_effect(metadata: pd.DataFrame, scp: pd.DataFrame):
    """Return a side_effect callable for pd.read_csv that dispatches by path."""

    def _mock(path, **kwargs):
        if "ptbxl_database" in str(path):
            return metadata.copy()
        return scp.copy()

    return _mock


@pytest.fixture
def csv_mocks():
    """Patch pd.read_csv to return fake PTB-XL data."""
    meta = _make_metadata()
    scp = _make_scp_df()
    side_effect = _csv_side_effect(meta, scp)
    with patch("ecg_ml_stream.dataset.ecg_dataset.pd.read_csv", side_effect=side_effect):
        yield meta, scp


@pytest.fixture
def csv_mocks_multi_patient():
    """Patch pd.read_csv so that patient 1001 has two records in the train split."""
    meta = pd.DataFrame(
        {
            "scp_codes": ["{'NORM': 100.0}"] * 5,
            "strat_fold": [1, 1, 1, 9, 10],
            "filename_lr": [f"records100/{i:05d}_lr" for i in range(1, 6)],
            "filename_hr": [f"records500/{i:05d}_hr" for i in range(1, 6)],
            "age": [50, 51, 52, 53, 54],
            "sex": [0, 1, 0, 1, 0],
            "patient_id": [1001, 1001, 1002, 1003, 1004],
        },
        index=pd.Index(range(1, 6), name="ecg_id"),
    )
    scp = _make_scp_df()
    with patch("ecg_ml_stream.dataset.ecg_dataset.pd.read_csv", side_effect=_csv_side_effect(meta, scp)):
        yield meta, scp


@pytest.fixture
def fake_signal_100hz() -> np.ndarray:
    """Fake wfdb signal: (1000 samples, 12 channels)."""
    return np.zeros((1000, 12), dtype=np.float32)


@pytest.fixture
def fake_signal_500hz() -> np.ndarray:
    """Fake wfdb signal: (5000 samples, 12 channels)."""
    return np.zeros((5000, 12), dtype=np.float32)
