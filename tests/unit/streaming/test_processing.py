"""Unit tests for processing module.

Copyright 2026 Mateusz Golebiewski
"""

import sys
from unittest.mock import patch

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
