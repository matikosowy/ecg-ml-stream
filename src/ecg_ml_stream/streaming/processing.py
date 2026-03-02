"""Spark processing module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import argparse

import pyspark.sql.functions as F  # noqa: N812 - common Spark convention
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StringType,
)

from ecg_ml_stream.streaming.udf import create_inference_udf
from ecg_ml_stream.utils.helpers import setup_logging
from ecg_ml_stream.utils.mappings import (
    DIAGNOSED_STREAM_RENAME,
    DIAGNOSED_STREAM_SELECT,
    PARSED_STREAM_RENAME,
)
from ecg_ml_stream.utils.schemas import STREAM_INPUT_SCHEMA

logger = setup_logging(name="streaming")


def run_streaming_job(
    kafka_bootstrap_servers: str = "kafka:9092",
    input_topic: str = "ecg-pending",
    output_topic: str = "ecg-diagnoses",
    model_path: str = "/app/models/ecg_resnet1d.pt",
    checkpoint_path: str = "/app/checkpoints/ecg-streaming",
) -> None:
    """Start the ECG Spark Structured Streaming job.

    Args:
        kafka_bootstrap_servers (str): Kafka's broker address.
        input_topic (str): Kafka topic to read ECG data from.
        output_topic (str): Kafka topic to write diagnoses to.
        model_path (str): Path to the ECG classification model.
        checkpoint_path (str): Path for Spark Structured Streaming checkpointing.

    """
    logger.info("=" * 60)
    logger.info(
        "ECG-ML-STREAM: Distributed real-time ECG signal processing system with ML capabilities."
    )
    logger.info("=" * 60)
    logger.info("Kafka:              %s", kafka_bootstrap_servers)
    logger.info("Input topic:        %s", input_topic)
    logger.info("Output topic:       %s", output_topic)
    logger.info("Model path:         %s", model_path)
    logger.info("Checkpoint path:    %s", checkpoint_path)
    logger.info("=" * 60)

    spark_builder = (
        SparkSession.builder.appName("ECG-ML-STREAM")
        .config("spark.sql.streaming.checkpointLocation", checkpoint_path)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.executor.memory", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.cores", "2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )

    with spark_builder.getOrCreate() as spark:
        spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark version:  %s", spark.version)
        logger.info("Spark master:   %s", spark.sparkContext.master)

        raw_stream = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
            .option("subscribe", input_topic)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .option("maxOffsetsPerTrigger", "100")
            .load()
        )

        logger.info("Kafka streaming job initialized!")

        parsed_stream = (
            raw_stream.withColumn(
                "key",
                F.col("key").cast(StringType()),
            )
            .withColumn(
                "json_value",
                F.col("value").cast(StringType()),
            )
            .withColumn(
                "data",
                F.from_json(F.col("json_value"), schema=STREAM_INPUT_SCHEMA),
            )
        )

        parsed_stream = parsed_stream.select(
            *[F.col(src).alias(dst) for src, dst in PARSED_STREAM_RENAME.items()]
        )

        inference_udf = create_inference_udf(model_path)
        diagnosed_stream = parsed_stream.withColumn(
            "diagnosis",
            inference_udf(F.col("signal_data"), F.col("sampling_rate")),
        ).withColumn(
            "timestamp_processed",
            F.current_timestamp(),
        )

        diagnosed_stream = diagnosed_stream.select(
            *[F.col(c) for c in DIAGNOSED_STREAM_SELECT],
            *[F.col(src).alias(dst) for src, dst in DIAGNOSED_STREAM_RENAME.items()],
        )

        output_stream = (
            diagnosed_stream.withColumn("value", F.to_json(F.struct("*")))
            .withColumnRenamed("exam_id", "key")
            .select("key", "value")
        )

        query = (
            output_stream.writeStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
            .option("topic", output_topic)
            .option("checkpointLocation", checkpoint_path)
            .outputMode("append")
            .trigger(processingTime="1 seconds")
            .start()
        )

        logger.info("Streaming job started! Waiting for data...")
        logger.info("Press Ctrl+C to stop.")

        try:
            query.awaitTermination()
        except KeyboardInterrupt:
            logger.info("Stopping streaming job...")
            query.stop()


def streaming_dry_run(
    kafka_bootstrap_servers: str = "kafka:9092",
    input_topic: str = "ecg-pending",
) -> None:
    """Run a debug dry-run of the streaming job without inference.

    Args:
        kafka_bootstrap_servers (str): Kafka's broker address.
        input_topic (str): Kafka topic to read ECG data from.

    """
    spark_builder = SparkSession.builder.appName("ECG-ML-STREAM-DEBUG")

    with spark_builder.getOrCreate() as spark:
        spark.sparkContext.setLogLevel("WARN")

        raw_stream = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
            .option("subscribe", input_topic)
            .option("startingOffsets", "latest")
            .load()
        )

        parsed_stream = (
            raw_stream.withColumn(
                "json_value",
                F.col("value").cast(StringType()),
            )
            .withColumn(
                "data",
                F.from_json(F.col("json_value"), schema=STREAM_INPUT_SCHEMA),
            )
            .select(
                F.col("data.exam_id"),
                F.col("data.hospital.name").alias("hospital"),
                F.col("data.patient.ecg_id"),
                F.col("data.metadata.ground_truth_name").alias("ground_truth"),
            )
        )

        query = (
            parsed_stream.writeStream.format("console")
            .outputMode("append")
            .option("truncate", "false")
            .trigger(processingTime="2 seconds")
            .start()
        )

        query.awaitTermination()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the streaming job.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(description="ECG-ML-STREAM Spark Structured Streaming Job")
    parser.add_argument("--kafka", type=str, default="kafka:9092", help="Kafka broker address.")
    parser.add_argument(
        "--input-topic", type=str, default="ecg-pending", help="Kafka topic to read ECG data from."
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default="ecg-diagnoses",
        help="Kafka topic to write diagnoses to.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/app/models/ecg_resnet1d.pt",
        help="Path to the ECG classification model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/app/checkpoints/ecg-streaming",
        help="Path for Spark Structured Streaming checkpointing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a debug dry-run of the streaming job without inference.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the streaming job."""
    args = parse_args()

    if args.dry_run:
        streaming_dry_run(
            kafka_bootstrap_servers=args.kafka,
            input_topic=args.input_topic,
        )
    else:
        run_streaming_job(
            kafka_bootstrap_servers=args.kafka,
            input_topic=args.input_topic,
            output_topic=args.output_topic,
            model_path=args.model_path,
            checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
