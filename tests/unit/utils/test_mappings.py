"""Unit tests for mappings module.

Copyright 2026 Mateusz Golebiewski
"""

from ecg_ml_stream.utils.mappings import (
    DIAGNOSED_STREAM_RENAME,
    DIAGNOSED_STREAM_SELECT,
    PARSED_STREAM_RENAME,
)

_PARSED_EXPECTED: dict[str, str] = {
    "key": "exam_id",
    "data.timestamp_sent": "timestamp_sent",
    "data.hospital": "hospital",
    "data.patient": "patient",
    "data.signal.data": "signal_data",
    "data.signal.sampling_rate": "sampling_rate",
    "data.signal.leads": "leads",
    "data.metadata": "metadata",
}

_DIAGNOSED_EXPECTED: dict[str, str] = {
    "diagnosis.diagnosis_class": "diagnosis_class",
    "diagnosis.diagnosis_class_idx": "diagnosis_class_idx",
    "diagnosis.diagnosis_probability": "diagnosis_probability",
    "diagnosis.all_probabilities": "all_probabilities",
    "diagnosis.is_dangerous": "is_dangerous",
    "diagnosis.diagnosis_description": "diagnosis_description",
    "diagnosis.processing_time_ms": "processing_time_ms",
}

_SELECT_EXPECTED: list[str] = [
    "exam_id",
    "timestamp_sent",
    "timestamp_processed",
    "hospital",
    "patient",
    "metadata",
    "signal_data",
    "sampling_rate",
    "leads",
]


class TestParsedStreamRename:
    def test_exact_mapping(self):
        assert PARSED_STREAM_RENAME == _PARSED_EXPECTED

    def test_maps_key_to_exam_id(self):
        assert PARSED_STREAM_RENAME["key"] == "exam_id"

    def test_maps_signal_data(self):
        assert PARSED_STREAM_RENAME["data.signal.data"] == "signal_data"

    def test_maps_sampling_rate(self):
        assert PARSED_STREAM_RENAME["data.signal.sampling_rate"] == "sampling_rate"

    def test_maps_hospital(self):
        assert PARSED_STREAM_RENAME["data.hospital"] == "hospital"

    def test_maps_patient(self):
        assert PARSED_STREAM_RENAME["data.patient"] == "patient"

    def test_maps_timestamp(self):
        assert PARSED_STREAM_RENAME["data.timestamp_sent"] == "timestamp_sent"

    def test_maps_metadata(self):
        assert PARSED_STREAM_RENAME["data.metadata"] == "metadata"

    def test_maps_leads(self):
        assert PARSED_STREAM_RENAME["data.signal.leads"] == "leads"

    def test_has_eight_entries(self):
        assert len(PARSED_STREAM_RENAME) == 8

    def test_all_values_are_strings(self):
        assert all(isinstance(v, str) for v in PARSED_STREAM_RENAME.values())

    def test_no_duplicate_values(self):
        values = list(PARSED_STREAM_RENAME.values())
        assert len(values) == len(set(values))


class TestDiagnosedStreamRename:
    def test_exact_mapping(self):
        assert DIAGNOSED_STREAM_RENAME == _DIAGNOSED_EXPECTED

    def test_maps_diagnosis_class(self):
        assert DIAGNOSED_STREAM_RENAME["diagnosis.diagnosis_class"] == "diagnosis_class"

    def test_maps_diagnosis_class_idx(self):
        assert DIAGNOSED_STREAM_RENAME["diagnosis.diagnosis_class_idx"] == "diagnosis_class_idx"

    def test_maps_diagnosis_probability(self):
        assert DIAGNOSED_STREAM_RENAME["diagnosis.diagnosis_probability"] == "diagnosis_probability"

    def test_maps_all_probabilities(self):
        assert DIAGNOSED_STREAM_RENAME["diagnosis.all_probabilities"] == "all_probabilities"

    def test_maps_is_dangerous(self):
        assert DIAGNOSED_STREAM_RENAME["diagnosis.is_dangerous"] == "is_dangerous"

    def test_maps_description(self):
        assert DIAGNOSED_STREAM_RENAME["diagnosis.diagnosis_description"] == "diagnosis_description"

    def test_maps_processing_time(self):
        assert DIAGNOSED_STREAM_RENAME["diagnosis.processing_time_ms"] == "processing_time_ms"

    def test_has_seven_entries(self):
        assert len(DIAGNOSED_STREAM_RENAME) == 7

    def test_all_values_are_strings(self):
        assert all(isinstance(v, str) for v in DIAGNOSED_STREAM_RENAME.values())

    def test_no_duplicate_values(self):
        values = list(DIAGNOSED_STREAM_RENAME.values())
        assert len(values) == len(set(values))


class TestDiagnosedStreamSelect:
    def test_exact_list(self):
        assert sorted(DIAGNOSED_STREAM_SELECT) == sorted(_SELECT_EXPECTED)

    def test_contains_exam_id(self):
        assert "exam_id" in DIAGNOSED_STREAM_SELECT

    def test_contains_signal_data(self):
        assert "signal_data" in DIAGNOSED_STREAM_SELECT

    def test_contains_sampling_rate(self):
        assert "sampling_rate" in DIAGNOSED_STREAM_SELECT

    def test_contains_hospital(self):
        assert "hospital" in DIAGNOSED_STREAM_SELECT

    def test_contains_patient(self):
        assert "patient" in DIAGNOSED_STREAM_SELECT

    def test_contains_metadata(self):
        assert "metadata" in DIAGNOSED_STREAM_SELECT

    def test_contains_timestamp_sent(self):
        assert "timestamp_sent" in DIAGNOSED_STREAM_SELECT

    def test_contains_leads(self):
        assert "leads" in DIAGNOSED_STREAM_SELECT

    def test_has_nine_entries(self):
        assert len(DIAGNOSED_STREAM_SELECT) == 9

    def test_no_duplicates(self):
        assert len(DIAGNOSED_STREAM_SELECT) == len(set(DIAGNOSED_STREAM_SELECT))

    def test_all_strings(self):
        assert all(isinstance(v, str) for v in DIAGNOSED_STREAM_SELECT)
