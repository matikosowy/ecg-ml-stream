"""Configuration package for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from ecg_ml_stream.config.settings import (
    AppConfig,
    DashboardConfig,
    DataConfig,
    KafkaConfig,
    LoggingConfig,
    ModelConfig,
    ProducerConfig,
    SparkConfig,
    TrainingConfig,
    load_config,
)

cfg: AppConfig = load_config()  # noqa: RUF067 - singleton must live in __init__

__all__ = [
    "AppConfig",
    "DashboardConfig",
    "DataConfig",
    "KafkaConfig",
    "LoggingConfig",
    "ModelConfig",
    "ProducerConfig",
    "SparkConfig",
    "TrainingConfig",
    "cfg",
    "load_config",
]
