"""Unit tests for config settings module in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import pytest

from ecg_ml_stream.config.settings import _cast, load_config


class TestCast:
    @pytest.mark.parametrize(("value", "expected"), [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ])
    def test_bool(self, value, expected):
        assert _cast(value, bool) is expected

    def test_int(self):
        assert _cast("42", int) == 42

    def test_float(self):
        assert _cast("3.14", float) == pytest.approx(3.14)

    def test_str_passthrough(self):
        assert _cast("kafka:9092", str) == "kafka:9092"


class TestEnvOverrides:
    def test_ecg_env_var_overrides_toml(self, monkeypatch):
        monkeypatch.setenv("ECG_KAFKA_NUM_PARTITIONS", "8")
        cfg = load_config()
        assert cfg.kafka.num_partitions == 8

    def test_ecg_env_var_overrides_string_field(self, monkeypatch):
        monkeypatch.setenv("ECG_KAFKA_BOOTSTRAP_SERVERS", "broker:9092")
        cfg = load_config()
        assert cfg.kafka.bootstrap_servers == "broker:9092"

    def test_ecg_env_var_overrides_float_field(self, monkeypatch):
        monkeypatch.setenv("ECG_TRAINING_LR", "0.01")
        cfg = load_config()
        assert cfg.training.lr == pytest.approx(0.01)


class TestLegacyAlias:
    def test_alias_applies_when_ecg_var_absent(self, monkeypatch):
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "legacy:9092")
        monkeypatch.delenv("ECG_KAFKA_BOOTSTRAP_SERVERS", raising=False)
        cfg = load_config()
        assert cfg.kafka.bootstrap_servers == "legacy:9092"

    def test_ecg_var_takes_priority_over_alias(self, monkeypatch):
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "legacy:9092")
        monkeypatch.setenv("ECG_KAFKA_BOOTSTRAP_SERVERS", "new:9092")
        cfg = load_config()
        assert cfg.kafka.bootstrap_servers == "new:9092"
