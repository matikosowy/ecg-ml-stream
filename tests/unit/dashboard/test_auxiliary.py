"""Unit tests for dashboard auxiliary module.

Copyright 2026 Mateusz Golebiewski
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from kafka import TopicPartition
from kafka.errors import KafkaError

from ecg_ml_stream.dashboard.auxiliary import (
    get_kafka_consumer,
    get_topic_message_count,
    parse_diagnosis_message,
)

_AUXILIARY = "ecg_ml_stream.dashboard.auxiliary"


def _mock_consumer(topic="topic", partitions=None, committed_offset=None):
    """Create a mock KafkaConsumer with partition and offset stubs."""
    if partitions is None:
        partitions = {0}
    mock = MagicMock()
    mock.partitions_for_topic.return_value = partitions

    tps = [TopicPartition(topic, p) for p in sorted(partitions)]
    mock.committed.return_value = committed_offset
    mock.end_offsets.return_value = dict.fromkeys(tps, 200)
    mock.beginning_offsets.return_value = dict.fromkeys(tps, 0)
    return mock


class TestGetTopicMessageCount:
    def test_returns_total_count_single_partition(self):
        mock = MagicMock()
        mock.partitions_for_topic.return_value = {0}
        tp = TopicPartition("t", 0)
        mock.end_offsets.return_value = {tp: 50}
        mock.beginning_offsets.return_value = {tp: 0}

        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            assert get_topic_message_count("localhost:9092", "t") == 50

    def test_sums_across_partitions(self):
        mock = MagicMock()
        mock.partitions_for_topic.return_value = {0, 1}
        tp0, tp1 = TopicPartition("t", 0), TopicPartition("t", 1)
        mock.end_offsets.return_value = {tp0: 100, tp1: 50}
        mock.beginning_offsets.return_value = {tp0: 10, tp1: 0}

        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            assert get_topic_message_count("localhost:9092", "t") == 140

    def test_accounts_for_beginning_offset(self):
        mock = MagicMock()
        mock.partitions_for_topic.return_value = {0}
        tp = TopicPartition("t", 0)
        mock.end_offsets.return_value = {tp: 80}
        mock.beginning_offsets.return_value = {tp: 30}

        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            assert get_topic_message_count("localhost:9092", "t") == 50

    def test_returns_zero_when_no_partitions(self):
        mock = MagicMock()
        mock.partitions_for_topic.return_value = set()

        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            assert get_topic_message_count("localhost:9092", "t") == 0

    def test_returns_zero_when_partitions_none(self):
        mock = MagicMock()
        mock.partitions_for_topic.return_value = None

        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            assert get_topic_message_count("localhost:9092", "t") == 0

    def test_closes_consumer(self):
        mock = MagicMock()
        mock.partitions_for_topic.return_value = {0}
        tp = TopicPartition("t", 0)
        mock.end_offsets.return_value = {tp: 10}
        mock.beginning_offsets.return_value = {tp: 0}

        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            get_topic_message_count("localhost:9092", "t")
        mock.close.assert_called_once()

    def test_returns_none_on_kafka_error(self):
        with patch(f"{_AUXILIARY}.KafkaConsumer", side_effect=KafkaError):
            assert get_topic_message_count("localhost:9092", "t") is None


class TestGetKafkaConsumer:
    @pytest.fixture(autouse=True)
    def _clear_session_group_id(self):
        """Ensure each test gets a fresh session group ID."""
        st.session_state.pop("_kafka_group_id", None)

    def test_returns_consumer_on_success(self):
        mock = _mock_consumer(committed_offset=50)
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            result = get_kafka_consumer("localhost:9092", "topic", "group-1")
        assert result is mock

    def test_passes_bootstrap_servers(self):
        mock = _mock_consumer(committed_offset=50)
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock) as mock_cls:
            get_kafka_consumer("broker:9092", "topic", "group-1")
        assert mock_cls.call_args[1]["bootstrap_servers"] == "broker:9092"

    def test_passes_group_id_with_base_prefix(self):
        mock = _mock_consumer(committed_offset=50)
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock) as mock_cls:
            get_kafka_consumer("localhost:9092", "topic", "my-group")
        group_id = mock_cls.call_args[1]["group_id"]
        assert group_id.startswith("my-group-")

    def test_auto_offset_reset_is_latest(self):
        mock = _mock_consumer(committed_offset=50)
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock) as mock_cls:
            get_kafka_consumer("localhost:9092", "topic", "group-1")
        assert mock_cls.call_args[1]["auto_offset_reset"] == "latest"

    def test_assigns_partitions(self):
        mock = _mock_consumer(topic="ecg-diag", committed_offset=50)
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            get_kafka_consumer("localhost:9092", "ecg-diag", "group-1")
        mock.assign.assert_called_once()
        assigned = mock.assign.call_args[0][0]
        assert all(isinstance(tp, TopicPartition) for tp in assigned)

    def test_seeks_on_new_session(self):
        mock = _mock_consumer(committed_offset=None)
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            get_kafka_consumer("localhost:9092", "topic", "g")
        mock.seek.assert_called_once()
        target = mock.seek.call_args[0][1]
        assert target == 0  # seek to beginning offset

    def test_seeks_to_beginning_offset(self):
        mock = MagicMock()
        mock.partitions_for_topic.return_value = {0}
        tp = TopicPartition("topic", 0)
        mock.committed.return_value = None
        mock.end_offsets.return_value = {tp: 30}
        mock.beginning_offsets.return_value = {tp: 20}

        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            get_kafka_consumer("localhost:9092", "topic", "g")
        target = mock.seek.call_args[0][1]
        assert target == 20  # always seek to beginning_offsets value

    def test_no_seek_when_offsets_committed(self):
        mock = _mock_consumer(committed_offset=180)
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            get_kafka_consumer("localhost:9092", "topic", "group-1")
        mock.seek.assert_not_called()

    def test_returns_none_when_no_partitions(self):
        mock = MagicMock()
        mock.partitions_for_topic.return_value = set()
        with patch(f"{_AUXILIARY}.KafkaConsumer", return_value=mock):
            result = get_kafka_consumer("localhost:9092", "topic", "group-1")
        assert result is None
        mock.close.assert_called_once()

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
