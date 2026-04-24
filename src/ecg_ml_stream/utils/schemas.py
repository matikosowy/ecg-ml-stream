"""Spark schemas module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

STREAM_INPUT_SCHEMA = StructType(
    [
        StructField("exam_id", StringType(), False),
        StructField("timestamp_sent", StringType(), False),
        StructField(
            "hospital",
            StructType(
                [
                    StructField("id", StringType(), True),
                    StructField("name", StringType(), True),
                    StructField("city", StringType(), True),
                ],
            ),
            True,
        ),
        StructField("thread_id", IntegerType(), True),
        StructField(
            "patient",
            StructType(
                [
                    StructField("patient_id", IntegerType(), True),
                    StructField("ecg_id", StringType(), True),
                    StructField("age", IntegerType(), True),
                    StructField("sex", StringType(), True),
                ],
            ),
            True,
        ),
        StructField(
            "signal",
            StructType(
                [
                    StructField("data", ArrayType(ArrayType(DoubleType())), True),
                    StructField("sampling_rate", IntegerType(), True),
                    StructField("num_channels", IntegerType(), True),
                    StructField("duration_seconds", DoubleType(), True),
                    StructField("leads", ArrayType(StringType()), True),
                ],
            ),
            True,
        ),
        StructField(
            "metadata",
            StructType(
                [
                    StructField("ground_truth_label", IntegerType(), True),
                    StructField("ground_truth_name", StringType(), True),
                ],
            ),
            True,
        ),
    ],
)


INFERENCE_OUTPUT_SCHEMA = StructType(
    [
        StructField("diagnosis_class", StringType(), True),
        StructField("diagnosis_class_idx", IntegerType(), True),
        StructField("diagnosis_probability", DoubleType(), True),
        StructField("all_probabilities", StringType(), True),
        StructField("is_dangerous", BooleanType(), True),
        StructField("diagnosis_description", StringType(), True),
        StructField("processing_time_ms", DoubleType(), True),
    ]
)
