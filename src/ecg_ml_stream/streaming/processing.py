"""Spark processing module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import argparse

import pyspark.sql.functions as F  # noqa: N812 - common Spark convention
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    StringType,
)

from ecg_ml_stream.config import cfg
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
    kafka_bootstrap_servers: str | None = None,
    input_topic: str | None = None,
    output_topic: str | None = None,
    model_path: str | None = None,
    checkpoint_path: str | None = None,
) -> None:
    """Start the ECG Spark Structured Streaming job.

    Args:
        kafka_bootstrap_servers (str): Kafka's broker address.
        input_topic (str): Kafka topic to read ECG data from.
        output_topic (str): Kafka topic to write diagnoses to.
        model_path (str): Path to the ECG classification model.
        checkpoint_path (str): Path for Spark Structured Streaming checkpointing.

    """
    kafka_bootstrap_servers = kafka_bootstrap_servers or cfg.kafka.bootstrap_servers
    input_topic = input_topic or cfg.kafka.topic_pending
    output_topic = output_topic or cfg.kafka.topic_diagnoses
    model_path = model_path or cfg.model.path
    checkpoint_path = checkpoint_path or cfg.spark.checkpoint_path

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
        SparkSession.builder.appName(cfg.spark.app_name)
        .config("spark.sql.streaming.checkpointLocation", checkpoint_path)
        .config("spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions))
        .config("spark.default.parallelism", str(cfg.spark.parallelism))
        .config("spark.executor.memory", cfg.spark.executor_memory)
        .config("spark.executor.cores", str(cfg.spark.executor_cores))
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .config("spark.driver.cores", str(cfg.spark.driver_cores))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )

    with spark_builder.getOrCreate() as spark:
        spark.sparkContext.setLogLevel("ERROR")
        logger.info("Spark version:  %s", spark.version)
        logger.info("Spark master:   %s", spark.sparkContext.master)

        raw_stream = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
            .option("subscribe", input_topic)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .option("maxOffsetsPerTrigger", str(cfg.spark.max_offsets_per_trigger))
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

        def _write_batch(batch_df: DataFrame, batch_id: int) -> None:
            """Log per-record diagnosis info and write the batch to Kafka."""
            rows = batch_df.select(
                "exam_id",
                "diagnosis_class",
                "diagnosis_probability",
                "is_dangerous",
                "processing_time_ms",
            ).collect()

            if not rows:
                return

            logger.info("Batch %d — %d record(s)", batch_id, len(rows))
            for row in rows:
                danger_tag = "  [DANGEROUS]" if row.is_dangerous else ""
                logger.info(
                    "  [%s]  %-6s  p=%.3f  %.0fms%s",
                    row.exam_id,
                    row.diagnosis_class or "ERROR",
                    row.diagnosis_probability or 0.0,
                    row.processing_time_ms or 0.0,
                    danger_tag,
                )

            (
                batch_df.withColumn("value", F.to_json(F.struct("*")))
                .withColumnRenamed("exam_id", "key")
                .select("key", "value")
                .write.format("kafka")
                .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
                .option("topic", output_topic)
                .save()
            )

        query = (
            diagnosed_stream.writeStream.foreachBatch(_write_batch)
            .option("checkpointLocation", checkpoint_path)
            .outputMode("append")
            .trigger(processingTime=cfg.spark.trigger_interval)
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
    kafka_bootstrap_servers: str | None = None,
    input_topic: str | None = None,
) -> None:
    """Run a debug dry-run of the streaming job without inference.

    Args:
        kafka_bootstrap_servers (str): Kafka's broker address.
        input_topic (str): Kafka topic to read ECG data from.

    """
    kafka_bootstrap_servers = kafka_bootstrap_servers or cfg.kafka.bootstrap_servers
    input_topic = input_topic or cfg.kafka.topic_pending

    spark_builder = SparkSession.builder.appName(f"{cfg.spark.app_name}-DEBUG")

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
    parser.add_argument(
        "--kafka", type=str, default=cfg.kafka.bootstrap_servers, help="Kafka broker address."
    )
    parser.add_argument(
        "--input-topic",
        type=str,
        default=cfg.kafka.topic_pending,
        help="Kafka topic to read ECG data from.",
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default=cfg.kafka.topic_diagnoses,
        help="Kafka topic to write diagnoses to.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=cfg.model.path,
        help="Path to the ECG classification model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=cfg.spark.checkpoint_path,
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
