"""Column mappings module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

PARSED_STREAM_RENAME = {
    "key": "exam_id",
    "data.timestamp_sent": "timestamp_sent",
    "data.hospital": "hospital",
    "data.patient": "patient",
    "data.signal.data": "signal_data",
    "data.signal.sampling_rate": "sampling_rate",
    "data.signal.leads": "leads",
    "data.metadata": "metadata",
}

DIAGNOSED_STREAM_RENAME = {
    "diagnosis.diagnosis_class": "diagnosis_class",
    "diagnosis.diagnosis_class_idx": "diagnosis_class_idx",
    "diagnosis.diagnosis_probability": "diagnosis_probability",
    "diagnosis.all_probabilities": "all_probabilities",
    "diagnosis.is_dangerous": "is_dangerous",
    "diagnosis.diagnosis_description": "diagnosis_description",
    "diagnosis.processing_time_ms": "processing_time_ms",
}


DIAGNOSED_STREAM_SELECT = [
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

CLASS_TRANSLATIONS = {
    "Normal sinus rhythm": "Prawidłowy rytm zatokowy",
    "Myocardial infarction": "Zawał mięśnia sercowego",
    "ST/T-wave changes": "Zmiany odcinka ST i załamka T",
    "Conduction disturbance": "Zaburzenia przewodzenia",
    "Cardiac hypertrophy": "Przerost serca",
}
