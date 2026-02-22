"""Unit tests for ECGProducer.

Copyright 2026 Mateusz Golebiewski
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from kafka.errors import KafkaError

from ecg_ml_stream.producer.ecg_producer import ECGProducer, main

_KAFKA_PRODUCER = "ecg_ml_stream.producer.ecg_producer.KafkaProducer"
_ECG_DATASET = "ecg_ml_stream.producer.ecg_producer.ECGDataset"
_TIME_SLEEP = "ecg_ml_stream.producer.ecg_producer.time.sleep"
_THREAD_POOL = "ecg_ml_stream.producer.ecg_producer.ThreadPoolExecutor"


class TestProducerMessage:
    def test_required_keys(self, sample_kafka_message):
        required = {
            "exam_id",
            "timestamp_sent",
            "hospital",
            "thread_id",
            "patient",
            "signal",
            "metadata",
        }
        assert required <= set(sample_kafka_message.keys())

    def test_signal_shape(self, sample_kafka_message):
        data = sample_kafka_message["signal"]["data"]
        assert len(data) == 12
        assert len(data[0]) == 1000

    def test_hospital_fields(self, sample_kafka_message):
        hospital = sample_kafka_message["hospital"]
        assert "id" in hospital
        assert "name" in hospital
        assert "city" in hospital

    def test_metadata_ground_truth(self, sample_kafka_message):
        meta = sample_kafka_message["metadata"]
        assert "ground_truth_label" in meta
        assert "ground_truth_name" in meta

    def test_exam_id_is_string(self, sample_kafka_message):
        assert isinstance(sample_kafka_message["exam_id"], str)

    def test_signal_sampling_rate(self, sample_kafka_message):
        assert sample_kafka_message["signal"]["sampling_rate"] == 100

    def test_signal_leads_count(self, sample_kafka_message):
        leads = sample_kafka_message["signal"]["leads"]
        assert len(leads) == 12


class TestECGProducerInit:
    def test_initial_state(self, producer):
        assert producer.running is False
        assert producer.stats["sent"] == 0
        assert producer.stats["errors"] == 0
        assert producer.stats["start_time"] is None

    def test_kafka_producer_receives_bootstrap_servers(self):
        with patch(_KAFKA_PRODUCER) as mock_kafka, patch(_ECG_DATASET):
            ECGProducer(bootstrap_servers="broker:9092", data_path="/fake")
        assert mock_kafka.call_args.kwargs["bootstrap_servers"] == "broker:9092"

    def test_dataset_initialized_with_test_split(self):
        with patch(_KAFKA_PRODUCER), patch(_ECG_DATASET) as mock_ds:
            ECGProducer(data_path="/fake", sampling_rate=500)
        mock_ds.assert_called_once_with(data_path="/fake", sampling_rate=500, split="test")


class TestCreateMessage:
    def test_message_has_required_keys(self, producer):
        msg = producer._create_message(thread_id=0)
        assert {
            "exam_id",
            "timestamp_sent",
            "hospital",
            "thread_id",
            "patient",
            "signal",
            "metadata",
        } <= set(msg)

    def test_thread_id_in_message(self, producer):
        msg = producer._create_message(thread_id=3)
        assert msg["thread_id"] == 3

    def test_exam_id_is_unique(self, producer):
        msg1 = producer._create_message(thread_id=0)
        msg2 = producer._create_message(thread_id=0)
        assert msg1["exam_id"] != msg2["exam_id"]

    @pytest.mark.parametrize(("sex_value", "expected"), [(0, "M"), (1, "F"), (99, None)])
    def test_sex_mapping(self, producer, fake_sample, sex_value, expected):
        fake_sample["sex"] = sex_value
        msg = producer._create_message(thread_id=0)
        assert msg["patient"]["sex"] == expected

    def test_signal_metadata(self, producer):
        msg = producer._create_message(thread_id=0)
        assert msg["signal"]["sampling_rate"] == producer.sampling_rate
        assert msg["signal"]["num_channels"] == 12
        assert msg["signal"]["duration_sec"] == 10.0
        assert len(msg["signal"]["leads"]) == 12

    def test_hospital_is_from_hospitals_list(self, producer):
        msg = producer._create_message(thread_id=0)
        hospital_ids = {h["id"] for h in ECGProducer.HOSPITALS}
        assert msg["hospital"]["id"] in hospital_ids


class TestProducerThread:
    def _run_one_iteration(self, producer: ECGProducer) -> None:
        """Run _producer_thread for exactly one iteration by stopping on sleep."""

        def stop_after_sleep(*args, **kwargs):
            producer.running = False

        producer.running = True
        with patch(_TIME_SLEEP, side_effect=stop_after_sleep):
            producer._producer_thread(thread_id=0)

    def test_successful_send_increments_sent(self, producer):
        mock_meta = MagicMock()
        mock_meta.partition = 0
        producer.producer.send.return_value.get.return_value = mock_meta

        self._run_one_iteration(producer)

        assert producer.stats["sent"] == 1
        assert producer.stats["errors"] == 0

    def test_kafka_error_increments_errors(self, producer):
        producer.producer.send.side_effect = KafkaError

        self._run_one_iteration(producer)

        assert producer.stats["errors"] == 1
        assert producer.stats["sent"] == 0

    def test_generic_exception_increments_errors(self, producer):
        producer.producer.send.side_effect = RuntimeError("unexpected")

        self._run_one_iteration(producer)

        assert producer.stats["errors"] == 1
        assert producer.stats["sent"] == 0


class TestStats:
    def test_stop_sets_running_false(self, producer):
        producer.running = True
        producer.stop()
        assert producer.running is False

    def test_print_stats_does_not_raise(self, producer):
        producer.stats["start_time"] = time.time()
        producer.stats["sent"] = 5
        producer.stats["errors"] = 1
        producer._print_stats()


class TestStart:
    def test_start_submits_threads_and_closes(self, producer):
        def fake_sleep(*args, **kwargs):
            producer.running = False

        with patch(_THREAD_POOL) as mock_tp, patch(_TIME_SLEEP, side_effect=fake_sleep):
            mock_executor = mock_tp.return_value.__enter__.return_value
            producer.start()

        mock_executor.submit.assert_called_once_with(producer._producer_thread, thread_id=0)
        producer.producer.close.assert_called_once()

    def test_start_handles_keyboard_interrupt(self, producer):
        def raise_interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        with patch(_THREAD_POOL) as mock_tp, patch(_TIME_SLEEP, side_effect=raise_interrupt):
            mock_tp.return_value.__enter__.return_value = MagicMock()
            producer.start()

        assert producer.running is False
        producer.producer.close.assert_called_once()


class TestMain:
    def test_main_exits_when_path_missing(self):
        with (
            patch("sys.argv", ["prog", "--data-path", "/nonexistent"]),
            patch("ecg_ml_stream.producer.ecg_producer.Path.exists", return_value=False),
        ):
            main()

    def test_main_creates_producer_and_starts(self):
        with (
            patch("sys.argv", ["prog", "--data-path", "/fake"]),
            patch("ecg_ml_stream.producer.ecg_producer.Path.exists", return_value=True),
            patch(_KAFKA_PRODUCER),
            patch(_ECG_DATASET),
            patch.object(ECGProducer, "start"),
        ):
            main()

    def test_main_handles_keyboard_interrupt(self):
        with (
            patch("sys.argv", ["prog", "--data-path", "/fake"]),
            patch("ecg_ml_stream.producer.ecg_producer.Path.exists", return_value=True),
            patch(_KAFKA_PRODUCER),
            patch(_ECG_DATASET),
            patch.object(ECGProducer, "start", side_effect=KeyboardInterrupt),
            patch.object(ECGProducer, "stop") as mock_stop,
        ):
            main()
        mock_stop.assert_called_once()
