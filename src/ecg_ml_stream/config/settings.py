"""Global configuration loader for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import os
import tomllib
from dataclasses import dataclass
from importlib import resources
from typing import Any


@dataclass
class KafkaConfig:
    """Kafka broker and topic settings."""

    bootstrap_servers: str
    topic_pending: str
    topic_diagnoses: str
    num_partitions: int
    replication_factor: int
    consumer_group: str
    consumer_timeout_ms: int


@dataclass
class DataConfig:
    """Dataset paths and signal parameters."""

    path: str
    sampling_rate: int
    window_size_sec: float
    window_stride_sec: float
    duration_sec: float
    num_leads: int


@dataclass
class ModelConfig:
    """Neural network architecture parameters."""

    path: str
    base_filters: int
    kernel_size: int
    num_blocks: list[int]
    dropout: float


@dataclass
class SparkConfig:
    """Spark cluster and streaming settings."""

    app_name: str
    shuffle_partitions: int
    parallelism: int
    executor_memory: str
    executor_cores: int
    driver_memory: str
    driver_cores: int
    checkpoint_path: str
    trigger_interval: str
    max_offsets_per_trigger: int


@dataclass
class DashboardConfig:
    """Streamlit dashboard settings."""

    port: int
    refresh_interval: int
    max_records: int
    max_stored: int
    default_leads: list[int]


@dataclass
class ProducerConfig:
    """Kafka producer thread settings."""

    num_threads: int
    interval_sec: float


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    num_workers: int
    seed: int
    output_dir: str
    grad_clip_norm: float
    scheduler_factor: float
    scheduler_patience: int
    voting_eval_interval: int
    voting_eval_samples: int


@dataclass
class LoggingConfig:
    """Logging settings."""

    dir: str
    format: str


@dataclass
class AppConfig:
    """Root configuration container for the entire application."""

    kafka: KafkaConfig
    data: DataConfig
    model: ModelConfig
    spark: SparkConfig
    dashboard: DashboardConfig
    producer: ProducerConfig
    training: TrainingConfig
    logging: LoggingConfig


_SECTION_CLASSES: dict[str, type] = {
    "kafka": KafkaConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "spark": SparkConfig,
    "dashboard": DashboardConfig,
    "producer": ProducerConfig,
    "training": TrainingConfig,
    "logging": LoggingConfig,
}

_ENV_ALIASES: dict[str, tuple[str, str]] = {
    "KAFKA_BOOTSTRAP_SERVERS": ("kafka", "bootstrap_servers"),
}


def _cast(value: str, target_type: type) -> Any:  # noqa: ANN401 - dynamic casting
    """Cast a string environment variable to the target field type.

    Returns:
        Converted value matching the target type.

    """
    if target_type is bool:
        return value.lower() in {"true", "1", "yes"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _apply_env_overrides(raw: dict[str, dict[str, Any]]) -> None:
    """Apply environment variable overrides to the parsed TOML dict.

    Convention: ECG_{SECTION}_{KEY} in uppercase.
    Example: ECG_KAFKA_BOOTSTRAP_SERVERS -> kafka.bootstrap_servers
    """
    for section_name, section_cls in _SECTION_CLASSES.items():
        section_dict = raw.setdefault(section_name, {})
        hints = section_cls.__dataclass_fields__

        for field_name, field_obj in hints.items():
            env_key = f"ECG_{section_name}_{field_name}".upper()
            env_val = os.environ.get(env_key)
            if env_val is not None:
                section_dict[field_name] = _cast(env_val, field_obj.type)

    for env_name, (section_name, key) in _ENV_ALIASES.items():
        env_val = os.environ.get(env_name)
        if env_val is not None:
            ecg_key = f"ECG_{section_name}_{key}".upper()
            if os.environ.get(ecg_key) is None:
                raw.setdefault(section_name, {})[key] = env_val


def load_config() -> AppConfig:
    """Load configuration from the bundled defaults.toml, then apply env overrides.

    Returns:
        AppConfig: Fully resolved application configuration.

    """
    config_pkg = resources.files("ecg_ml_stream.config")
    toml_bytes = (config_pkg / "defaults.toml").read_bytes()
    raw = tomllib.loads(toml_bytes.decode())

    _apply_env_overrides(raw)

    sections = {}
    for section_name, section_cls in _SECTION_CLASSES.items():
        section_data = raw.get(section_name, {})
        sections[section_name] = section_cls(**section_data)

    return AppConfig(**sections)
