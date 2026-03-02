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


def _make_series(
    n_rows: int = 1,
    sampling_rate: int = 100,
    *,
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

    def test_custom_model_path_used(self, fake_diagnosis):
        custom_path = "/custom/model.pt"
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis) as mock_infer:
            _call_udf(signal_s, rate_s, model_path=custom_path)
        assert mock_infer.call_args[1]["model_path"] == custom_path


class TestInferenceUdfSuccess:
    def test_returns_dataframe(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert isinstance(result, pd.DataFrame)

    def test_result_has_one_row(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert len(result) == 1

    def test_diagnosis_class_correct(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_class"] == "NORM"

    def test_diagnosis_class_idx_correct(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_class_idx"] == 0

    def test_diagnosis_probability_correct(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_probability"] == pytest.approx(0.9)

    def test_all_probabilities_is_json_string(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        parsed = json.loads(result.iloc[0]["all_probabilities"])
        assert parsed["NORM"] == pytest.approx(0.9)

    def test_is_dangerous_false(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert not result.iloc[0]["is_dangerous"]

    def test_diagnosis_description_correct(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["diagnosis_description"] == "Normal sinus rhythm"

    def test_processing_time_ms_non_negative(self, fake_diagnosis):
        signal_s, rate_s = _make_series()
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0]["processing_time_ms"] >= 0.0

    def test_multiple_rows(self, fake_diagnosis):
        signal_s, rate_s = _make_series(n_rows=3)
        with patch(_INFER, return_value=fake_diagnosis):
            result = _call_udf(signal_s, rate_s)
        assert len(result) == 3

    def test_signal_data_as_json_string_is_parsed(self, fake_diagnosis):
        signal_s, rate_s = _make_series(signal_as_json=True)
        with patch(_INFER, return_value=fake_diagnosis) as mock_infer:
            _call_udf(signal_s, rate_s)
        call_kwargs = mock_infer.call_args[1]
        assert isinstance(call_kwargs["signal_data"], list)

    def test_sampling_rate_forwarded_as_int(self, fake_diagnosis):
        signal_s, rate_s = _make_series(sampling_rate=500)
        with patch(_INFER, return_value=fake_diagnosis) as mock_infer:
            _call_udf(signal_s, rate_s)
        assert mock_infer.call_args[1]["sampling_rate"] == 500


class TestInferenceUdfError:
    @pytest.mark.parametrize(
        "field",
        [
            "diagnosis_class",
            "diagnosis_class_idx",
            "diagnosis_probability",
            "is_dangerous",
            "processing_time_ms",
        ],
    )
    def test_error_field_is_none(self, field):
        signal_s, rate_s = _make_series()
        with patch(_INFER, side_effect=RuntimeError):
            result = _call_udf(signal_s, rate_s)
        assert result.iloc[0][field] is None

    def test_error_description_contains_message(self):
        signal_s, rate_s = _make_series()
        with patch(_INFER, side_effect=RuntimeError("something went wrong")):
            result = _call_udf(signal_s, rate_s)
        assert "something went wrong" in result.iloc[0]["diagnosis_description"]

    def test_error_does_not_stop_other_rows(self, fake_diagnosis):
        signal_s, rate_s = _make_series(n_rows=3)
        with patch(_INFER, side_effect=[RuntimeError("fail"), fake_diagnosis, fake_diagnosis]):
            result = _call_udf(signal_s, rate_s)
        assert len(result) == 3
        assert result.iloc[0]["diagnosis_class"] is None
        assert result.iloc[1]["diagnosis_class"] == "NORM"
        assert result.iloc[2]["diagnosis_class"] == "NORM"
