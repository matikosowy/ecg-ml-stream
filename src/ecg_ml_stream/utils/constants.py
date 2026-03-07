"""Constants module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

CLASS_NAMES: list[str] = ["NORM", "MI", "STTC", "CD", "HYP"]

CLASS_DESCRIPTIONS: dict[str, str] = {
    "NORM": "Normal sinus rhythm",
    "MI": "Myocardial infarction",
    "STTC": "ST/T-wave changes",
    "CD": "Conduction disturbance",
    "HYP": "Cardiac hypertrophy",
}

DANGEROUS_CLASSES: list[str] = ["MI", "STTC", "CD", "HYP"]

SUPERCLASS_MAPPING: dict[str, int] = {
    "NORM": 0,
    "MI": 1,
    "STTC": 2,
    "CD": 3,
    "HYP": 4,
}
SUPERCLASS_PRIORITY: list[str] = ["MI", "STTC", "CD", "HYP", "NORM"]

NUM_CLASSES: int = 5
NUM_LEADS: int = 12

ECG_LEAD_NAMES: list[str] = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

CLASS_COLORS: dict[str, str] = {
    "NORM": "#4caf50",
    "MI": "#f44336",
    "STTC": "#ff9800",
    "CD": "#9c27b0",
    "HYP": "#2196f3",
}
