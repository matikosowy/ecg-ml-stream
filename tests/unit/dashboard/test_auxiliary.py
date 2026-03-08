"""Unit tests for dashboard auxiliary module.

Copyright 2026 Mateusz Golebiewski
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from kafka.errors import KafkaError

from ecg_ml_stream.dashboard.auxiliary import get_kafka_consumer, parse_diagnosis_message

_AUXILIARY = "ecg_ml_stream.dashboard.auxiliary"


class TestGetKafkaConsumer:
    @pytest.fixture(autouse=True)
    def _clear_session_group_id(self):
        """Ensure each test gets a fresh session group ID."""
        st.session_state.pop("_kafka_group_id", None)

    def test_returns_consumer_on_success(self):
        mock_consumer = MagicMock()
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock_consumer):
            result = get_kafka_consumer("localhost:9092", "test-topic", "group-1")
        assert result is mock_consumer

    def test_passes_topic_as_first_arg(self):
        with patch(f"{_AUXILIARY}.KafkaConsumer") as mock_cls:
            get_kafka_consumer("localhost:9092", "ecg-pending", "group-1")
        assert mock_cls.call_args[0][0] == "ecg-pending"

    def test_passes_bootstrap_servers(self):
        with patch(f"{_AUXILIARY}.KafkaConsumer") as mock_cls:
            get_kafka_consumer("broker:9092", "topic", "group-1")
        assert mock_cls.call_args[1]["bootstrap_servers"] == "broker:9092"

    def test_passes_group_id_with_base_prefix(self):
        with patch(f"{_AUXILIARY}.KafkaConsumer") as mock_cls:
            get_kafka_consumer("localhost:9092", "topic", "my-group")
        group_id = mock_cls.call_args[1]["group_id"]
        assert group_id.startswith("my-group-")

    def test_kafka_error_returns_none(self):
        with (
            patch(f"{_AUXILIARY}.KafkaConsumer", side_effect=KafkaError),
            patch(f"{_AUXILIARY}.st"),
        ):
            result = get_kafka_consumer("localhost:9092", "topic", "group")
        assert result is None

    def test_kafka_error_calls_st_error(self):
        with (
            patch(f"{_AUXILIARY}.KafkaConsumer", side_effect=KafkaError),
            patch(f"{_AUXILIARY}.st") as mock_st,
        ):
            get_kafka_consumer("localhost:9092", "topic", "group")
        mock_st.error.assert_called_once()


class TestParseDiagnosisMessage:
    def test_returns_dict_on_valid_message(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert isinstance(result, dict)

    def test_exam_id_extracted(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert result["exam_id"] == "abc-123"

    def test_hospital_fields_extracted(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert result["hospital_id"] == "H1"
        assert result["hospital_name"] == "Hospital A"
        assert result["hospital_city"] == "Warsaw"

    def test_patient_fields_extracted(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert result["patient_age"] == 45
        assert result["patient_sex"] == "M"
        assert result["patient_ecg_id"] == 42

    def test_diagnosis_fields_extracted(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert result["diagnosis_class"] == "NORM"
        assert result["diagnosis_probability"] == pytest.approx(0.9)
        assert result["is_dangerous"] is False

    def test_ground_truth_extracted(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert result["ground_truth"] == "NORM"

    def test_processing_time_extracted(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert result["processing_time_ms"] == pytest.approx(100.0)

    def test_all_probabilities_as_dict_unchanged(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert result["all_probabilities"] == fake_message["all_probabilities"]

    def test_all_probabilities_as_json_string_parsed(self, fake_message):
        probs = {"NORM": 0.9, "MI": 0.1}
        fake_message["all_probabilities"] = json.dumps(probs)
        result = parse_diagnosis_message(fake_message)
        assert result["all_probabilities"] == probs

    def test_signal_data_as_list_unchanged(self, fake_message):
        result = parse_diagnosis_message(fake_message)
        assert result["signal_data"] == fake_message["signal_data"]

    def test_signal_data_as_json_string_parsed(self, fake_message):
        data = [[0.1, 0.2, 0.3]]
        fake_message["signal_data"] = json.dumps(data)
        result = parse_diagnosis_message(fake_message)
        assert result["signal_data"] == data

    def test_missing_keys_use_scalar_defaults(self):
        result = parse_diagnosis_message({})
        assert result["exam_id"] == "N/A"
        assert result["diagnosis_class"] == "N/A"
        assert result["diagnosis_probability"] == 0
        assert result["is_dangerous"] is False
        assert result["all_probabilities"] == {}
        assert result["processing_time_ms"] == 0
        assert result["sampling_rate"] == 100

    def test_missing_hospital_uses_defaults(self):
        result = parse_diagnosis_message({})
        assert result["hospital_id"] == "N/A"
        assert result["hospital_name"] == "N/A"
        assert result["hospital_city"] == "N/A"

    def test_missing_patient_fields_are_none(self):
        result = parse_diagnosis_message({})
        assert result["patient_age"] is None
        assert result["patient_sex"] is None
        assert result["patient_ecg_id"] is None

    def test_exception_returns_none(self):
        with patch(f"{_AUXILIARY}.st"):
            result = parse_diagnosis_message(None)
        assert result is None

    def test_exception_calls_st_error(self):
        with patch(f"{_AUXILIARY}.st") as mock_st:
            parse_diagnosis_message(None)
        mock_st.error.assert_called_once()
