"""Unit tests for helpers module.

Copyright 2026 Mateusz Golebiewski
"""

import numpy as np

from ecg.utils.helpers import normalize_signal


class TestNormalizeSignal:
    def test_zero_mean(self):
        signal = np.random.default_rng(0).standard_normal((12, 250)).astype(np.float32)
        normed = normalize_signal(signal)
        means = normed.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_std(self):
        signal = np.random.default_rng(0).standard_normal((12, 250)).astype(np.float32)
        normed = normalize_signal(signal)
        stds = normed.std(axis=-1)
        np.testing.assert_allclose(stds, 1.0, atol=1e-2)

    def test_constant_channel_no_nan(self):
        signal = np.ones((12, 250), dtype=np.float32)
        normed = normalize_signal(signal)
        assert not np.any(np.isnan(normed))
