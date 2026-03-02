"""Pandas UDF functions module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import json
from collections.abc import Iterator
from datetime import datetime

import pandas as pd
from pyspark.sql.functions import pandas_udf

from ecg_ml_stream.ml.inference import infer_ecg_record
from ecg_ml_stream.utils.schemas import STREAM_OUTPUT_SCHEMA


@pandas_udf(STREAM_OUTPUT_SCHEMA)
def inference_udf_function(
    iterator: Iterator[pd.DataFrame],
    model_path: str = "/app/models/ecg_resnet1d.pt",
) -> Iterator[pd.DataFrame]:
    """Run inference for every row in an Arrow batch.

    Args:
        iterator (Iterator[pd.DataFrame]): Iterator over input DataFrames.
        model_path (str): Path to the ECG classification model.

    Yields:
        Iterator[pd.DataFrame]: Iterator over DataFrames containing diagnosis results.

    """
    for batch_df in iterator:
        results = []

        for _, row in batch_df.iterrows():
            try:
                signal_data = row["signal_data"]
                if isinstance(signal_data, str):
                    signal_data = json.loads(signal_data)

                sampling_rate = row["sampling_rate"]
                processing_start = datetime.now()

                diagnosis = infer_ecg_record(
                    signal_data=signal_data,
                    sampling_rate=sampling_rate,
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

            except Exception as e:  # noqa: BLE001 - Allow broad exception handling to capture various inference errors
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

        yield pd.DataFrame(results)
