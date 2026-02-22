"""Fixtures for producer unit tests in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from collections.abc import Generator
from unittest.mock import patch

import numpy as np
import pytest

from ecg_ml_stream.producer.ecg_producer import ECGProducer

_KAFKA_PRODUCER = "ecg_ml_stream.producer.ecg_producer.KafkaProducer"
_ECG_DATASET = "ecg_ml_stream.producer.ecg_producer.ECGDataset"


@pytest.fixture
def sample_kafka_message() -> dict:
    """Return a fake Kafka message matching the ECGProducer format.

    Returns:
        dict: Dictionary with all required message fields.
    """
    rng = np.random.default_rng(42)
    return {
        "exam_id": "test-exam-001",
        "timestamp_sent": "2024-01-01",
        "hospital": {
            "id": "HOSP_001",
            "name": "Test Hospital",
            "city": "Testcity",
        },
        "thread_id": 0,
        "patient": {"ecg_id": 1, "age": 55, "sex": "M"},
        "signal": {
            "data": rng.standard_normal((12, 1000)).tolist(),
            "sampling_rate": 100,
            "num_channels": 12,
            "duration_sec": 10.0,
            "leads": [
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ],
        },
        "metadata": {"ground_truth_label": 0, "ground_truth_name": "NORM"},
    }


@pytest.fixture
def fake_sample() -> dict:
    """Return a fake ECG sample as returned by ECGDataset.get_sample_for_streaming.

    Returns:
        dict: Sample dict with signal data and metadata.
    """
    rng = np.random.default_rng(42)
    return {
        "ecg_id": 1,
        "signal": rng.standard_normal((12, 1000)).tolist(),
        "label": 0,
        "label_name": "NORM",
        "age": 55,
        "sex": 0,
    }


@pytest.fixture
def producer(fake_sample: dict) -> Generator[ECGProducer, None, None]:
    """Return an ECGProducer with mocked Kafka and dataset dependencies.

    Returns:
        ECGProducer: Producer instance ready for testing.
    """
    with patch(_KAFKA_PRODUCER), patch(_ECG_DATASET) as mock_ds_cls:
        mock_ds = mock_ds_cls.return_value
        mock_ds.get_sample_for_streaming.return_value = fake_sample
        yield ECGProducer(data_path="/fake", num_threads=1, interval_sec=0.1)
