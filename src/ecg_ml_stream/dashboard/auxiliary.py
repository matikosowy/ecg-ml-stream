"""Auxiliary module for Streamlit dashboard app for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from kafka import KafkaConsumer
from kafka.errors import KafkaError

import streamlit as st
import json


def get_kafka_consumer(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
) -> KafkaConsumer | None:
    """Create a Kafka consumer for the specified topic.
    
    Args:
        bootstrap_servers (str): Comma-separated list of Kafka bootstrap servers.
        topic (str): Kafka topic to subscribe to.
        group_id (str): Optional. Kafka consumer group ID.

    Returns:
        KafkaConsumer | None: A KafkaConsumer instance if successful, or None if an error occurs.
    
    """
    try:
        return KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id=group_id,
            consumer_timeout_ms=500,
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
            "sampling_rate": message.get("sampling_rate", 100),
        }
    except Exception as e:
        st.error(f"Message parsing error: {e}")
        return None
