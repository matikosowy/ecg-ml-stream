"""Unit tests for processing module.

Copyright 2026 Mateusz Golebiewski
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from ecg_ml_stream.streaming.processing import (
    main,
    parse_args,
    run_streaming_job,
    streaming_dry_run,
)

_PROCESSING = "ecg_ml_stream.streaming.processing"


@pytest.fixture
def streaming_env(mock_spark_builder, mock_spark, mock_query):
    """Patch SparkSession and pyspark.sql.functions for run_streaming_job tests."""
    with (
        patch(f"{_PROCESSING}.SparkSession") as mock_cls,
        patch(f"{_PROCESSING}.F"),
    ):
        mock_cls.builder = mock_spark_builder
        yield mock_spark, mock_query


@pytest.fixture
def dry_run_env(mock_spark_builder, mock_spark, mock_query):
    """Patch SparkSession and pyspark.sql.functions for streaming_dry_run tests."""
    with (
        patch(f"{_PROCESSING}.SparkSession") as mock_cls,
        patch(f"{_PROCESSING}.F"),
    ):
        mock_cls.builder = mock_spark_builder
        yield mock_spark, mock_query


class TestParseArgs:
    @pytest.mark.parametrize(
        ("argv", "attr", "expected"),
        [
            ([], "kafka", "kafka:9092"),
            (["--kafka", "broker:1234"], "kafka", "broker:1234"),
            ([], "input_topic", "ecg-pending"),
            (["--input-topic", "my-input"], "input_topic", "my-input"),
            ([], "output_topic", "ecg-diagnoses"),
            (["--output-topic", "my-output"], "output_topic", "my-output"),
            ([], "model_path", "/app/models/ecg_resnet1d.pt"),
            (["--model-path", "/models/model.pt"], "model_path", "/models/model.pt"),
            ([], "checkpoint", "/app/checkpoints/ecg-streaming"),
            (["--checkpoint", "/checkpoints/ckpt"], "checkpoint", "/checkpoints/ckpt"),
            ([], "dry_run", False),
            (["--dry-run"], "dry_run", True),
        ],
    )
    def test_argument(self, argv, attr, expected):
        with patch.object(sys, "argv", ["prog", *argv]):
            args = parse_args()
        assert getattr(args, attr) == expected


class TestMain:
    def test_normal_mode_calls_run_streaming_job(self):
        with (
            patch.object(sys, "argv", ["prog"]),
            patch(f"{_PROCESSING}.run_streaming_job") as mock_job,
            patch(f"{_PROCESSING}.streaming_dry_run") as mock_dry,
        ):
            main()
        mock_job.assert_called_once()
        mock_dry.assert_not_called()

    def test_dry_run_mode_calls_streaming_dry_run(self):
        with (
            patch.object(sys, "argv", ["prog", "--dry-run"]),
            patch(f"{_PROCESSING}.run_streaming_job") as mock_job,
            patch(f"{_PROCESSING}.streaming_dry_run") as mock_dry,
        ):
            main()
        mock_dry.assert_called_once()
        mock_job.assert_not_called()

    @pytest.mark.parametrize(
        ("argv", "kwarg", "expected"),
        [
            (["--kafka", "b:9999"], "kafka_bootstrap_servers", "b:9999"),
            (["--input-topic", "in-t"], "input_topic", "in-t"),
            (["--output-topic", "out-t"], "output_topic", "out-t"),
            (["--model-path", "/m.pt"], "model_path", "/m.pt"),
            (["--checkpoint", "/ckpt"], "checkpoint_path", "/ckpt"),
        ],
    )
    def test_args_forwarded_to_streaming_job(self, argv, kwarg, expected):
        with (
            patch.object(sys, "argv", ["prog", *argv]),
            patch(f"{_PROCESSING}.run_streaming_job") as mock_job,
        ):
            main()
        assert mock_job.call_args.kwargs[kwarg] == expected

    @pytest.mark.parametrize(
        ("argv", "kwarg", "expected"),
        [
            (["--kafka", "b:9999"], "kafka_bootstrap_servers", "b:9999"),
            (["--input-topic", "in-t"], "input_topic", "in-t"),
        ],
    )
    def test_args_forwarded_to_dry_run(self, argv, kwarg, expected):
        with (
            patch.object(sys, "argv", ["prog", "--dry-run", *argv]),
            patch(f"{_PROCESSING}.streaming_dry_run") as mock_dry,
        ):
            main()
        assert mock_dry.call_args.kwargs[kwarg] == expected


class TestRunStreamingJob:
    def test_spark_session_created(self, streaming_env, mock_spark_builder):
        with patch(f"{_PROCESSING}.create_inference_udf"):
            run_streaming_job()
        mock_spark_builder.getOrCreate.assert_called_once()

    def test_model_path_forwarded_to_udf(self, streaming_env):
        with patch(f"{_PROCESSING}.create_inference_udf") as mock_udf:
            run_streaming_job(model_path="/custom/model.pt")
        mock_udf.assert_called_once_with("/custom/model.pt")

    def test_awaits_query_termination(self, streaming_env, mock_query):
        with patch(f"{_PROCESSING}.create_inference_udf"):
            run_streaming_job()
        mock_query.awaitTermination.assert_called_once()

    def test_keyboard_interrupt_stops_query(self, streaming_env, mock_query):
        mock_query.awaitTermination.side_effect = KeyboardInterrupt
        with patch(f"{_PROCESSING}.create_inference_udf"):
            run_streaming_job()
        mock_query.stop.assert_called_once()

    def test_kafka_bootstrap_servers_passed_to_read(self, streaming_env, mock_spark):
        with patch(f"{_PROCESSING}.create_inference_udf"):
            run_streaming_job(kafka_bootstrap_servers="test-broker:9092")
        option_values = [c[0][1] for c in mock_spark.readStream.option.call_args_list]
        assert "test-broker:9092" in option_values

    def test_input_topic_passed_to_read(self, streaming_env, mock_spark):
        with patch(f"{_PROCESSING}.create_inference_udf"):
            run_streaming_job(input_topic="my-ecg-topic")
        option_values = [c[0][1] for c in mock_spark.readStream.option.call_args_list]
        assert "my-ecg-topic" in option_values


class TestWriteBatch:
    """Tests for the _write_batch closure passed to foreachBatch."""

    def _get_write_batch_fn(self, streaming_env):
        mock_spark, _ = streaming_env
        mock_df = mock_spark.readStream.load.return_value
        return mock_df.writeStream.foreachBatch.call_args[0][0]

    def _make_batch_df(self, rows):
        select_result = MagicMock()
        select_result.collect.return_value = rows

        write_chain = MagicMock()
        write_chain.withColumnRenamed.return_value = write_chain
        write_chain.select.return_value = write_chain

        batch_df = MagicMock()
        batch_df.select.return_value = select_result
        batch_df.withColumn.return_value = write_chain
        return batch_df

    def test_empty_batch_skips_write(self, streaming_env):
        with patch(f"{_PROCESSING}.create_inference_udf"):
            run_streaming_job()
        write_batch = self._get_write_batch_fn(streaming_env)

        batch_df = self._make_batch_df([])
        write_batch(batch_df, 0)

        batch_df.withColumn.assert_not_called()

    def test_rows_written_to_kafka(self, streaming_env):
        with patch(f"{_PROCESSING}.create_inference_udf"):
            run_streaming_job()
        write_batch = self._get_write_batch_fn(streaming_env)

        row = MagicMock()
        row.is_dangerous = False
        row.exam_id = "abc-123"
        row.diagnosis_class = "NORM"
        row.diagnosis_probability = 0.9
        row.processing_time_ms = 100.0

        batch_df = self._make_batch_df([row])
        write_batch(batch_df, 1)

        write_chain = batch_df.withColumn.return_value
        write_chain.write.format.assert_called_with("kafka")

    def test_dangerous_row_sets_danger_tag(self, streaming_env):
        with patch(f"{_PROCESSING}.create_inference_udf"):
            run_streaming_job()
        write_batch = self._get_write_batch_fn(streaming_env)

        row = MagicMock()
        row.is_dangerous = True

        batch_df = self._make_batch_df([row])
        write_batch(batch_df, 2)

        batch_df.withColumn.assert_called_once()


class TestStreamingDryRun:
    def test_spark_session_created(self, dry_run_env, mock_spark_builder):
        streaming_dry_run()
        mock_spark_builder.getOrCreate.assert_called_once()

    def test_awaits_query_termination(self, dry_run_env, mock_query):
        streaming_dry_run()
        mock_query.awaitTermination.assert_called_once()

    def test_kafka_bootstrap_servers_passed_to_read(self, dry_run_env, mock_spark):
        streaming_dry_run(kafka_bootstrap_servers="dry-broker:9092")
        option_values = [c[0][1] for c in mock_spark.readStream.option.call_args_list]
        assert "dry-broker:9092" in option_values

    def test_input_topic_passed_to_read(self, dry_run_env, mock_spark):
        streaming_dry_run(input_topic="dry-topic")
        option_values = [c[0][1] for c in mock_spark.readStream.option.call_args_list]
        assert "dry-topic" in option_values
