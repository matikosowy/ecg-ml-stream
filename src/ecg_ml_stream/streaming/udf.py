"""Pandas UDF functions module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import json
from datetime import datetime

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.udf import UserDefinedFunction

from ecg_ml_stream.config import cfg
from ecg_ml_stream.ml.inference import infer_ecg_record
from ecg_ml_stream.utils.schemas import INFERENCE_OUTPUT_SCHEMA


def create_inference_udf(
    model_path: str | None = None,
) -> UserDefinedFunction:
    """Return a pandas UDF that runs ECG inference with the given model path.

    Args:
        model_path (str): Path to the ECG classification model.

    Returns:
        UserDefinedFunctionLike: Registered pandas UDF ready for use in Spark.

    """
    model_path = model_path or cfg.model.path

    @pandas_udf(INFERENCE_OUTPUT_SCHEMA)
    def _udf(
        signal_data_series: pd.Series,
        sampling_rate_series: pd.Series,
    ) -> pd.DataFrame:
        results = []

        for signal_data, sampling_rate in zip(
            signal_data_series, sampling_rate_series, strict=True
        ):
            try:
                parsed = json.loads(signal_data) if isinstance(signal_data, str) else signal_data

                processing_start = datetime.now()

                diagnosis = infer_ecg_record(
                    signal_data=parsed,
                    sampling_rate=int(sampling_rate),
                    model_path=model_path,
                )

                processing_end = datetime.now()
                processing_time_ms = (processing_end - processing_start).total_seconds() * 1000

                results.append(
                    {
                        "diagnosis_class": diagnosis["class"],
                        "diagnosis_class_idx": diagnosis["class_idx"],
                        "diagnosis_probability": diagnosis["probability"],
                        "all_probabilities": json.dumps(diagnosis["all_probabilities"]),
                        "is_dangerous": diagnosis["is_dangerous"],
                        "diagnosis_description": diagnosis["description"],
                        "processing_time_ms": processing_time_ms,
                    }
                )

            except Exception as e:  # noqa: BLE001 - Capture various inference errors per row
                results.append(
                    {
                        "diagnosis_class": None,
                        "diagnosis_class_idx": None,
                        "diagnosis_probability": None,
                        "all_probabilities": None,
                        "is_dangerous": None,
                        "diagnosis_description": str(e),
                        "processing_time_ms": None,
                    }
                )

        return pd.DataFrame(results)

    return _udf
