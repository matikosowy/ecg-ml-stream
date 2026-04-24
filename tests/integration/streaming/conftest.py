"""Fixtures for streaming integration tests in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from ecg_ml_stream.utils.mappings import PARSED_STREAM_RENAME
from ecg_ml_stream.utils.schemas import STREAM_INPUT_SCHEMA

_KAFKA_RAW_SCHEMA = StructType(
    [
        StructField("key", BinaryType()),
        StructField("value", BinaryType()),
        StructField("topic", StringType()),
        StructField("partition", IntegerType()),
        StructField("offset", LongType()),
        StructField("timestamp", TimestampType()),
        StructField("timestampType", IntegerType()),
    ]
)


@pytest.fixture(scope="module")
def sample_payload() -> dict:
    """Return a realistic ECG Kafka payload matching STREAM_INPUT_SCHEMA."""
    rng = np.random.default_rng(42)
    signal_data = rng.standard_normal((12, 1000)).tolist()
    return {
        "exam_id": "test-exam-uuid-1234",
        "timestamp_sent": "2026-01-01T10:00:00",
        "hospital": {"id": "H1", "name": "Szpital Testowy", "city": "Warszawa"},
        "thread_id": 1,
        "patient": {"patient_id": 13619, "ecg_id": "ECG001", "age": 45, "sex": "M"},
        "signal": {
            "data": signal_data,
            "sampling_rate": 100,
            "num_channels": 12,
            "duration_seconds": 10.0,
            "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        },
        "metadata": {"ground_truth_label": 0, "ground_truth_name": "NORM"},
    }


@pytest.fixture(scope="module")
def raw_kafka_df(spark: SparkSession, sample_payload: dict) -> DataFrame:
    """Return a static DataFrame mimicking kafka readStream output with one ECG row."""
    value_bytes = bytearray(json.dumps(sample_payload).encode("utf-8"))
    key_bytes = bytearray(sample_payload["exam_id"].encode("utf-8"))
    rows = [
        {
            "key": key_bytes,
            "value": value_bytes,
            "topic": "ecg-pending",
            "partition": 0,
            "offset": 0,
            "timestamp": datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC),
            "timestampType": 0,
        }
    ]
    return spark.createDataFrame(rows, schema=_KAFKA_RAW_SCHEMA)


@pytest.fixture
def bad_kafka_df(spark: SparkSession) -> DataFrame:
    """Return a DataFrame with an unparseable JSON value."""
    rows = [
        {
            "key": bytearray(b"k"),
            "value": bytearray(b"not-valid-json"),
            "topic": "ecg-pending",
            "partition": 0,
            "offset": 0,
            "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
            "timestampType": 0,
        }
    ]
    return spark.createDataFrame(rows, schema=_KAFKA_RAW_SCHEMA)


@pytest.fixture(scope="module")
def parsed_df(raw_kafka_df: DataFrame) -> DataFrame:
    """Return DataFrame after cast, from_json and select — mirrors run_streaming_job logic."""
    df = (
        raw_kafka_df.withColumn("key", F.col("key").cast(StringType()))
        .withColumn("json_value", F.col("value").cast(StringType()))
        .withColumn("data", F.from_json(F.col("json_value"), schema=STREAM_INPUT_SCHEMA))
    )
    return df.select(*[F.col(src).alias(dst) for src, dst in PARSED_STREAM_RENAME.items()])
