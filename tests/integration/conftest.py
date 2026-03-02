"""Fixtures for integration tests in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from collections.abc import Generator

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Create a local SparkSession shared across the entire integration test session."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("ecg-integration-test")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def fake_diagnosis() -> dict:
    """Return a realistic inference result dict."""
    return {
        "class": "NORM",
        "class_idx": 0,
        "probability": 0.9,
        "all_probabilities": {"NORM": 0.9, "MI": 0.025, "STTC": 0.025, "CD": 0.025, "HYP": 0.025},
        "is_dangerous": False,
        "description": "Normal sinus rhythm",
    }
