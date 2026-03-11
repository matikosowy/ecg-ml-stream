"""Auxiliary module for Streamlit dashboard app for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import json
import logging
import uuid

import streamlit as st
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError

from ecg_ml_stream.config import cfg

logger = logging.getLogger(__name__)


def _session_group_id(base: str) -> str:
    """Return a consumer group ID unique to the current Streamlit session.

    A fresh ID is generated on every new session (page refresh), so the
    consumer starts without committed offsets and seeks to recent messages.
    Within a session the ID stays the same, so auto-refresh only fetches
    new messages via committed offsets.

    Args:
        base (str): Base prefix for the group ID.

    Returns:
        str: A string like ``dashboard-consumer-<uuid4_hex[:8]>``.

    """
    key = "_kafka_group_id"
    if key not in st.session_state:
        st.session_state[key] = f"{base}-{uuid.uuid4().hex[:8]}"
    return st.session_state[key]


def get_topic_message_count(bootstrap_servers: str, topic: str) -> int | None:
    """Return the total number of messages in a Kafka topic.

    Uses beginning/end offset metadata — no messages are consumed.
    Returns ``None`` on any connection or metadata error so callers can
    distinguish "0 messages" from "Kafka unreachable".

    Args:
        bootstrap_servers (str): Comma-separated list of Kafka bootstrap servers.
        topic (str): Kafka topic name.

    Returns:
        int | None: Total message count across all partitions, or None on error.

    """
    try:
        consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            consumer.close()
            return 0
        tps = [TopicPartition(topic, p) for p in partitions]
        end = consumer.end_offsets(tps)
        begin = consumer.beginning_offsets(tps)
        consumer.close()
        return sum(end[tp] - begin[tp] for tp in tps)
    except KafkaError:
        logger.debug("Cannot query topic offsets for %s", topic)
        return None


def get_kafka_consumer(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    max_backfill: int = 100,
) -> KafkaConsumer | None:
    """Create a Kafka consumer positioned to read recent messages.

    On a new session (no committed offsets) the consumer seeks to at most
    ``max_backfill`` messages from the end of each partition, avoiding a
    full replay of the topic.  On subsequent auto-refresh cycles the
    committed offsets are reused so only new messages are fetched.

    Args:
        bootstrap_servers (str): Comma-separated list of Kafka bootstrap servers.
        topic (str): Kafka topic to subscribe to.
        group_id (str): Base Kafka consumer group ID (suffixed per session).
        max_backfill (int): Maximum number of recent messages to load on a
            fresh session.  Defaults to 100.

    Returns:
        KafkaConsumer | None: A KafkaConsumer instance if successful, or None.

    """
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id=_session_group_id(group_id),
            consumer_timeout_ms=cfg.kafka.consumer_timeout_ms,
        )

        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            consumer.close()
            return None

        tps = [TopicPartition(topic, p) for p in partitions]
        consumer.assign(tps)

        # New session: no committed offsets — seek to recent messages only
        if all(consumer.committed(tp) is None for tp in tps):
            end_offsets = consumer.end_offsets(tps)
            begin_offsets = consumer.beginning_offsets(tps)
            per_partition = max(1, max_backfill // len(tps))
            for tp in tps:
                target = max(begin_offsets[tp], end_offsets[tp] - per_partition)
                consumer.seek(tp, target)
    except KafkaError as e:
        st.error(f"Kafka connection error: {e}")
        return None
    else:
        return consumer


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
