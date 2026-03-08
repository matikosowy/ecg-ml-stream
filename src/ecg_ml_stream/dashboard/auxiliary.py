"""Auxiliary module for Streamlit dashboard app for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import json
import uuid

import streamlit as st
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from ecg_ml_stream.config import cfg


def _session_group_id(base: str) -> str:
    """Return a consumer group ID unique to the current Streamlit session.

    Uses ``st.session_state`` to persist the generated ID across reruns
    within the same browser session, while producing a fresh ID on page
    refresh (new session).

    Args:
        base: Base prefix for the group ID.

    Returns:
        A string like `dashboard-consumer-<uuid4_hex[:8]>`.

    """
    key = "_kafka_group_id"
    if key not in st.session_state:
        st.session_state[key] = f"{base}-{uuid.uuid4().hex[:8]}"
    return st.session_state[key]


def get_kafka_consumer(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
) -> KafkaConsumer | None:
    """Create a Kafka consumer for the specified topic.

    On the first run of a new Streamlit session the consumer reads from
    the earliest available offset so that historical messages are loaded.
    Within the same session subsequent polls only fetch new messages.

    Args:
        bootstrap_servers (str): Comma-separated list of Kafka bootstrap servers.
        topic (str): Kafka topic to subscribe to.
        group_id (str): Base Kafka consumer group ID (suffixed per session).

    Returns:
        KafkaConsumer | None: A KafkaConsumer instance if successful, or None if an error occurs.

    """
    try:
        return KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id=_session_group_id(group_id),
            consumer_timeout_ms=cfg.kafka.consumer_timeout_ms,
        )
    except KafkaError as e:
        st.error(f"Kafka connection error: {e}")
        return None


def parse_diagnosis_message(message: dict) -> dict | None:
    """Normalize a raw Kafka diagnosis message into a flat dictionary format.

    Args:
        message (dict): The raw Kafka message containing ECG diagnosis data.

    Returns:
        dict | None: Normalized diagnosis dictionary, or None if parsing fails.

    """
    try:
        all_probs = message.get("all_probabilities")
        if isinstance(all_probs, str):
            all_probs = json.loads(all_probs)

        signal_data = message.get("signal_data")
        if isinstance(signal_data, str):
            signal_data = json.loads(signal_data)

        return {
            "exam_id": message.get("exam_id", "N/A"),
            "timestamp_sent": message.get("timestamp_sent"),
            "timestamp_processed": message.get("timestamp_processed"),
            "hospital_id": message.get("hospital", {}).get("id", "N/A"),
            "hospital_name": message.get("hospital", {}).get("name", "N/A"),
            "hospital_city": message.get("hospital", {}).get("city", "N/A"),
            "patient_age": message.get("patient", {}).get("age"),
            "patient_sex": message.get("patient", {}).get("sex"),
            "patient_ecg_id": message.get("patient", {}).get("ecg_id"),
            "diagnosis_class": message.get("diagnosis_class", "N/A"),
            "diagnosis_probability": message.get("diagnosis_probability", 0),
            "all_probabilities": all_probs or {},
            "is_dangerous": message.get("is_dangerous", False),
            "diagnosis_description": message.get("diagnosis_description", ""),
            "processing_time_ms": message.get("processing_time_ms", 0),
            "ground_truth": message.get("metadata", {}).get("ground_truth_name"),
            "signal_data": signal_data,
            "sampling_rate": message.get("sampling_rate", cfg.data.sampling_rate),
        }
    except Exception as e:  # noqa: BLE001 - Catch all exceptions for parsing errors
        st.error(f"Message parsing error: {e}")
        return None
