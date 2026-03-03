"""Integration tests for the streaming processing pipeline.

Copyright 2026 Mateusz Golebiewski
"""

import json
from unittest.mock import patch

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructType

from ecg_ml_stream.streaming.udf import create_inference_udf
from ecg_ml_stream.utils.mappings import (
    DIAGNOSED_STREAM_RENAME,
    DIAGNOSED_STREAM_SELECT,
    PARSED_STREAM_RENAME,
)
from ecg_ml_stream.utils.schemas import INFERENCE_OUTPUT_SCHEMA, STREAM_INPUT_SCHEMA

_INFER = "ecg_ml_stream.streaming.udf.infer_ecg_record"


class TestStreamInputSchema:
    """Tests that STREAM_INPUT_SCHEMA correctly parses ECG Kafka payloads."""

    def test_from_json_parses_exam_id(self, raw_kafka_df):
        result = (
            raw_kafka_df.select(
                F.from_json(
                    F.col("value").cast(StringType()),
                    schema=__import__(
                        "ecg_ml_stream.utils.schemas", fromlist=["STREAM_INPUT_SCHEMA"]
                    ).STREAM_INPUT_SCHEMA,
                ).alias("data")
            )
            .select("data.exam_id")
            .first()[0]
        )
        assert result == "test-exam-uuid-1234"

    def test_from_json_parses_sampling_rate(self, raw_kafka_df):
        result = (
            raw_kafka_df.withColumn("json_value", F.col("value").cast(StringType()))
            .withColumn("data", F.from_json(F.col("json_value"), schema=STREAM_INPUT_SCHEMA))
            .select("data.signal.sampling_rate")
            .first()[0]
        )
        assert result == 100

    def test_from_json_parses_signal_data_shape(self, raw_kafka_df):
        result = (
            raw_kafka_df.withColumn("json_value", F.col("value").cast(StringType()))
            .withColumn("data", F.from_json(F.col("json_value"), schema=STREAM_INPUT_SCHEMA))
            .select("data.signal.data")
            .first()[0]
        )
        assert len(result) == 12
        assert len(result[0]) == 1000

    def test_from_json_invalid_json_returns_null(self, bad_kafka_df):
        result = (
            bad_kafka_df.withColumn("json_value", F.col("value").cast(StringType()))
            .withColumn("data", F.from_json(F.col("json_value"), schema=STREAM_INPUT_SCHEMA))
            .select("data.exam_id")
            .first()[0]
        )
        assert result is None

    def test_key_cast_to_string_matches_exam_id(self, raw_kafka_df, sample_payload):
        result = (
            raw_kafka_df.withColumn("key_str", F.col("key").cast(StringType()))
            .select("key_str")
            .first()[0]
        )
        assert result == sample_payload["exam_id"]


class TestParsedStreamTransformations:
    """Tests the full parse -> rename -> select pipeline on a static DataFrame."""

    def test_output_has_expected_columns(self, parsed_df):
        assert set(parsed_df.columns) == set(PARSED_STREAM_RENAME.values())

    def test_exam_id_value_correct(self, parsed_df):
        result = parsed_df.select("exam_id").first()[0]
        assert result == "test-exam-uuid-1234"

    def test_signal_data_column_is_array(self, parsed_df):
        field = next(f for f in parsed_df.schema.fields if f.name == "signal_data")
        assert isinstance(field.dataType, ArrayType)

    def test_sampling_rate_value_correct(self, parsed_df):
        result = parsed_df.select("sampling_rate").first()[0]
        assert result == 100

    def test_timestamp_sent_value_correct(self, parsed_df):
        result = parsed_df.select("timestamp_sent").first()[0]
        assert result == "2026-01-01T10:00:00"


class TestDiagnosedStreamTransformations:
    """Tests the diagnosed stream rename and Kafka output format."""

    @pytest.fixture
    def diagnosed_df(self, parsed_df: DataFrame) -> DataFrame:
        """Add a fake diagnosis struct and apply DIAGNOSED_STREAM_RENAME + select."""
        df = parsed_df.withColumn(
            "diagnosis",
            F.struct(
                F.lit("NORM").alias("diagnosis_class"),
                F.lit(0).alias("diagnosis_class_idx"),
                F.lit(0.9).alias("diagnosis_probability"),
                F.lit('{"NORM": 0.9}').alias("all_probabilities"),
                F.lit(False).alias("is_dangerous"),
                F.lit("Normal sinus rhythm").alias("diagnosis_description"),
                F.lit(5.0).alias("processing_time_ms"),
            ),
        ).withColumn("timestamp_processed", F.current_timestamp())

        return df.select(
            *[F.col(c) for c in DIAGNOSED_STREAM_SELECT],
            *[F.col(src).alias(dst) for src, dst in DIAGNOSED_STREAM_RENAME.items()],
        )

    def test_diagnosed_columns_present(self, diagnosed_df):
        expected = set(DIAGNOSED_STREAM_SELECT) | set(DIAGNOSED_STREAM_RENAME.values())
        assert set(diagnosed_df.columns) == expected

    def test_output_kafka_format(self, diagnosed_df):
        output = (
            diagnosed_df.withColumn("value", F.to_json(F.struct("*")))
            .withColumnRenamed("exam_id", "key")
            .select("key", "value")
        )
        assert output.columns == ["key", "value"]

    def test_value_is_valid_json(self, diagnosed_df):
        output = (
            diagnosed_df.withColumn("value", F.to_json(F.struct("*")))
            .withColumnRenamed("exam_id", "key")
            .select("value")
        )
        value = output.first()[0]
        parsed = json.loads(value)
        assert "diagnosis_class" in parsed


class TestCreateInferenceUdfIntegration:
    """Tests UDF registration and output schema with a real SparkSession."""

    def test_udf_return_type_matches_inference_schema(self):
        udf = create_inference_udf()
        assert udf.returnType == INFERENCE_OUTPUT_SCHEMA

    def test_udf_output_columns_on_dataframe(self, spark: SparkSession, fake_diagnosis):
        input_schema = (
            StructType()
            .add("signal_data", ArrayType(ArrayType(DoubleType())))
            .add("sampling_rate", IntegerType())
        )

        rows = [{"signal_data": [[0.1] * 10] * 12, "sampling_rate": 100}]
        df = spark.createDataFrame(rows, schema=input_schema)

        udf_fn = create_inference_udf()

        with patch(_INFER, return_value=fake_diagnosis):
            result = df.withColumn(
                "diagnosis", udf_fn(F.col("signal_data"), F.col("sampling_rate"))
            )

        output_fields = {f.name for f in result.schema["diagnosis"].dataType.fields}
        expected_fields = {f.name for f in INFERENCE_OUTPUT_SCHEMA.fields}
        assert output_fields == expected_fields
