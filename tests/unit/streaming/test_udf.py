"""Unit tests for UDF module.

Copyright 2026 Mateusz Golebiewski
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ecg_ml_stream.streaming.udf import create_inference_udf

_INFER = "ecg_ml_stream.streaming.udf.infer_ecg_record"
_DEFAULT_MODEL_PATH = "/app/models/ecg_resnet1d.pt"


def _fake_diagnosis() -> dict:
    return {
        "class": "NORM",
        "class_idx": 0,
        "probability": 0.9,
        "all_probabilities": {"NORM": 0.9, "MI": 0.025, "STTC": 0.025, "CD": 0.025, "HYP": 0.025},
        "is_dangerous": False,
        "description": "Normal sinus rhythm",
    }


def _make_series(
    n_rows: int = 1,
    sampling_rate: int = 100,
    signal_as_json: bool = False,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((12, 1000)).tolist()
    signal_col = json.dumps(signal) if signal_as_json else signal
    return (
        pd.Series([signal_col] * n_rows),
        pd.Series([sampling_rate] * n_rows),
    )


def _call_udf(
    signal_data: pd.Series,
    sampling_rate: pd.Series,
    model_path: str = _DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    udf_fn = create_inference_udf(model_path)
    return udf_fn.func(signal_data, sampling_rate)


class TestCreateInferenceUdf:
    def test_returns_callable(self):
        udf_fn = create_inference_udf()
        assert callable(udf_fn.func)

    def test_custom_model_path_used(self):
        custom_path = "/custom/model.pt"
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()) as mock_infer:
            _call_udf(signal_s, rate_s, model_path=custom_path)
        assert mock_infer.call_args[1]["model_path"] == custom_path


class TestInferenceUdfSuccess:
    def test_returns_dataframe(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert isinstance(result, pd.DataFrame)

    def test_result_has_one_row(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert len(result) == 1

    def test_diagnosis_class_correct(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_class"] == "NORM"

    def test_diagnosis_class_idx_correct(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_class_idx"] == 0

    def test_diagnosis_probability_correct(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_probability"] == pytest.approx(0.9)

    def test_all_probabilities_is_json_string(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        parsed = json.loads(result.iloc[0]["all_probabilities"])
        assert parsed["NORM"] == pytest.approx(0.9)

    def test_is_dangerous_false(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert not result.iloc[0]["is_dangerous"]

    def test_diagnosis_description_correct(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_description"] == "Normal sinus rhythm"

    def test_processing_time_ms_non_negative(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["processing_time_ms"] >= 0.0

    def test_multiple_rows(self):
        signal_s, rate_s = _make_series(n_rows=3)
        with patch(_INFER, return_value=_fake_diagnosis()):
            result = _call_udf(signal_s, rate_s)
        assert len(result) == 3

    def test_signal_data_as_json_string_is_parsed(self):
        signal_s, rate_s = _make_series(signal_as_json=True)
        with patch(_INFER, return_value=_fake_diagnosis()) as mock_infer:
            _call_udf(signal_s, rate_s)
        call_kwargs = mock_infer.call_args[1]
        assert isinstance(call_kwargs["signal_data"], list)

    def test_sampling_rate_forwarded_as_int(self):
        signal_s, rate_s = _make_series(sampling_rate=500)
        with patch(_INFER, return_value=_fake_diagnosis()) as mock_infer:
            _call_udf(signal_s, rate_s)
        assert mock_infer.call_args[1]["sampling_rate"] == 500


class TestInferenceUdfError:
    def test_error_diagnosis_class_is_none(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, side_effect=RuntimeError("model failed")):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_class"] is None

    def test_error_class_idx_is_none(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, side_effect=ValueError("bad input")):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_class_idx"] is None

    def test_error_probability_is_none(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, side_effect=RuntimeError):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_probability"] is None

    def test_error_is_dangerous_is_none(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, side_effect=RuntimeError):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["is_dangerous"] is None

    def test_error_description_contains_message(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, side_effect=RuntimeError("something went wrong")):
            result = _call_udf(signal_s, rate_s)
        assert "something went wrong" in result.iloc[0]["diagnosis_description"]

    def test_error_processing_time_is_none(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, side_effect=RuntimeError):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["processing_time_ms"] is None

    def test_error_does_not_stop_other_rows(self):
        signal_s, rate_s = _make_series(n_rows=3)
        diagnosis = _fake_diagnosis()
        with patch(_INFER, side_effect=[RuntimeError("fail"), diagnosis, diagnosis]):
            result = _call_udf(signal_s, rate_s)
        assert len(result) == 3
        assert result.iloc[0]["diagnosis_class"] is None
        assert result.iloc[1]["diagnosis_class"] == "NORM"
        assert result.iloc[2]["diagnosis_class"] == "NORM"
