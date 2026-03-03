"""Unit tests for schemas module.

Copyright 2026 Mateusz Golebiewski
"""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructType,
)

from ecg_ml_stream.utils.schemas import INFERENCE_OUTPUT_SCHEMA, STREAM_INPUT_SCHEMA


def _field(schema: StructType, name: str):
    return schema[name]


class TestStreamInputSchema:
    def test_is_struct_type(self):
        assert isinstance(STREAM_INPUT_SCHEMA, StructType)

    def test_top_level_field_names(self):
        names = {f.name for f in STREAM_INPUT_SCHEMA.fields}
        assert names == {
            "exam_id",
            "timestamp_sent",
            "hospital",
            "thread_id",
            "patient",
            "signal",
            "metadata",
        }

    def test_exam_id_not_nullable(self):
        assert _field(STREAM_INPUT_SCHEMA, "exam_id").nullable is False

    def test_timestamp_sent_not_nullable(self):
        assert _field(STREAM_INPUT_SCHEMA, "timestamp_sent").nullable is False

    def test_exam_id_is_string(self):
        assert isinstance(_field(STREAM_INPUT_SCHEMA, "exam_id").dataType, StringType)

    def test_thread_id_is_integer(self):
        assert isinstance(_field(STREAM_INPUT_SCHEMA, "thread_id").dataType, IntegerType)

    def test_hospital_is_struct(self):
        assert isinstance(_field(STREAM_INPUT_SCHEMA, "hospital").dataType, StructType)

    def test_hospital_nested_fields(self):
        hospital = _field(STREAM_INPUT_SCHEMA, "hospital").dataType
        names = {f.name for f in hospital.fields}
        assert names == {"id", "name", "city"}

    def test_hospital_nested_types_are_string(self):
        hospital = _field(STREAM_INPUT_SCHEMA, "hospital").dataType
        for f in hospital.fields:
            assert isinstance(f.dataType, StringType)

    def test_patient_is_struct(self):
        assert isinstance(_field(STREAM_INPUT_SCHEMA, "patient").dataType, StructType)

    def test_patient_nested_fields(self):
        patient = _field(STREAM_INPUT_SCHEMA, "patient").dataType
        names = {f.name for f in patient.fields}
        assert names == {"ecg_id", "age", "sex"}

    def test_patient_age_is_integer(self):
        patient = _field(STREAM_INPUT_SCHEMA, "patient").dataType
        assert isinstance(patient["age"].dataType, IntegerType)

    def test_signal_is_struct(self):
        assert isinstance(_field(STREAM_INPUT_SCHEMA, "signal").dataType, StructType)

    def test_signal_nested_fields(self):
        signal = _field(STREAM_INPUT_SCHEMA, "signal").dataType
        names = {f.name for f in signal.fields}
        assert names == {"data", "sampling_rate", "num_channels", "duration_seconds", "leads"}

    def test_signal_data_is_nested_array(self):
        signal = _field(STREAM_INPUT_SCHEMA, "signal").dataType
        data_field = signal["data"]
        assert isinstance(data_field.dataType, ArrayType)
        assert isinstance(data_field.dataType.elementType, ArrayType)
        assert isinstance(data_field.dataType.elementType.elementType, DoubleType)

    def test_signal_leads_is_string_array(self):
        signal = _field(STREAM_INPUT_SCHEMA, "signal").dataType
        leads = signal["leads"]
        assert isinstance(leads.dataType, ArrayType)
        assert isinstance(leads.dataType.elementType, StringType)

    def test_metadata_is_struct(self):
        assert isinstance(_field(STREAM_INPUT_SCHEMA, "metadata").dataType, StructType)

    def test_metadata_nested_fields(self):
        metadata = _field(STREAM_INPUT_SCHEMA, "metadata").dataType
        names = {f.name for f in metadata.fields}
        assert names == {"ground_truth_label", "ground_truth_name"}

    def test_metadata_ground_truth_label_is_integer(self):
        metadata = _field(STREAM_INPUT_SCHEMA, "metadata").dataType
        assert isinstance(metadata["ground_truth_label"].dataType, IntegerType)


class TestStreamOutputSchema:
    def test_is_struct_type(self):
        assert isinstance(INFERENCE_OUTPUT_SCHEMA, StructType)

    def test_field_names(self):
        names = {f.name for f in INFERENCE_OUTPUT_SCHEMA.fields}
        assert names == {
            "diagnosis_class",
            "diagnosis_class_idx",
            "diagnosis_probability",
            "all_probabilities",
            "is_dangerous",
            "diagnosis_description",
            "processing_time_ms",
        }

    def test_has_seven_fields(self):
        assert len(INFERENCE_OUTPUT_SCHEMA.fields) == 7

    def test_diagnosis_class_is_string(self):
        assert isinstance(_field(INFERENCE_OUTPUT_SCHEMA, "diagnosis_class").dataType, StringType)

    def test_diagnosis_class_idx_is_integer(self):
        assert isinstance(
            _field(INFERENCE_OUTPUT_SCHEMA, "diagnosis_class_idx").dataType, IntegerType
        )

    def test_diagnosis_probability_is_double(self):
        assert isinstance(
            _field(INFERENCE_OUTPUT_SCHEMA, "diagnosis_probability").dataType, DoubleType
        )

    def test_all_probabilities_is_string(self):
        assert isinstance(_field(INFERENCE_OUTPUT_SCHEMA, "all_probabilities").dataType, StringType)

    def test_is_dangerous_is_boolean(self):
        assert isinstance(_field(INFERENCE_OUTPUT_SCHEMA, "is_dangerous").dataType, BooleanType)

    def test_processing_time_ms_is_double(self):
        assert isinstance(
            _field(INFERENCE_OUTPUT_SCHEMA, "processing_time_ms").dataType, DoubleType
        )

    def test_all_fields_nullable(self):
        for f in INFERENCE_OUTPUT_SCHEMA.fields:
            assert f.nullable is True
