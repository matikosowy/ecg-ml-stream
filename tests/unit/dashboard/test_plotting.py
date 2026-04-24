"""Unit tests for dashboard plotting module.

Copyright 2026 Mateusz Golebiewski
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from ecg_ml_stream.dashboard.plotting import (
    create_ecg_plot,
    create_patient_exam_timeline,
    create_probability_chart,
)
from ecg_ml_stream.utils.constants import CLASS_COLORS, ECG_LEAD_NAMES


class TestCreateEcgPlot:
    def test_returns_figure(self, make_signal):
        fig = create_ecg_plot(make_signal(), sampling_rate=100)
        assert isinstance(fig, go.Figure)

    def test_default_selected_leads_uses_all(self, make_signal):
        fig = create_ecg_plot(make_signal(num_leads=12), sampling_rate=100)
        assert len(fig.data) == 12

    def test_selected_leads_limits_traces(self, make_signal):
        fig = create_ecg_plot(make_signal(), sampling_rate=100, selected_leads=[0, 1, 2])
        assert len(fig.data) == 3

    def test_trace_names_match_lead_names(self, make_signal):
        leads = [0, 3, 6]
        fig = create_ecg_plot(make_signal(), sampling_rate=100, selected_leads=leads)
        names = [trace.name for trace in fig.data]
        assert names == [ECG_LEAD_NAMES[i] for i in leads]

    def test_time_axis_length_matches_num_samples(self, make_signal):
        num_samples = 500
        signal = make_signal(num_samples=num_samples)
        fig = create_ecg_plot(signal, sampling_rate=100, selected_leads=[0])
        assert len(fig.data[0].x) == num_samples

    def test_time_axis_values_scaled_by_sampling_rate(self, make_signal):
        num_samples = 300
        sampling_rate = 150
        signal = make_signal(num_samples=num_samples)
        fig = create_ecg_plot(signal, sampling_rate=sampling_rate, selected_leads=[0])
        expected = np.arange(num_samples) / sampling_rate
        np.testing.assert_allclose(fig.data[0].x, expected)

    def test_height_minimum_is_400(self, make_signal):
        fig = create_ecg_plot(make_signal(), sampling_rate=100, selected_leads=[0])
        assert fig.layout.height == 400

    def test_height_scales_with_many_leads(self, make_signal):
        num_leads = 7
        signal = make_signal(num_leads=num_leads)
        fig = create_ecg_plot(signal, sampling_rate=100, selected_leads=list(range(num_leads)))
        assert fig.layout.height == max(400, 80 * num_leads)

    def test_out_of_bounds_lead_uses_empty_signal(self, make_signal):
        signal = make_signal(num_leads=2)
        fig = create_ecg_plot(signal, sampling_rate=100, selected_leads=[5])
        assert list(fig.data[0].y) == []

    @pytest.mark.parametrize("lead_idx", [0, 4, 11])
    def test_trace_y_matches_signal_data(self, make_signal, lead_idx):
        signal = make_signal(num_leads=12, num_samples=100)
        fig = create_ecg_plot(signal, sampling_rate=100, selected_leads=[lead_idx])
        assert list(fig.data[0].y) == signal[lead_idx]


class TestCreateProbabilityChart:
    def test_returns_figure(self):
        fig = create_probability_chart({"NORM": 0.9, "MI": 0.1})
        assert isinstance(fig, go.Figure)

    def test_probabilities_scaled_to_percent(self):
        fig = create_probability_chart({"NORM": 0.75})
        assert fig.data[0].y[0] == pytest.approx(75.0)

    def test_bar_count_equals_classes(self):
        probs = {"NORM": 0.5, "MI": 0.3, "CD": 0.2}
        fig = create_probability_chart(probs)
        assert len(fig.data[0].x) == 3

    def test_known_class_uses_class_color(self):
        fig = create_probability_chart({"NORM": 1.0})
        assert fig.data[0].marker.color[0] == CLASS_COLORS["NORM"]

    def test_unknown_class_uses_default_color(self):
        fig = create_probability_chart({"UNKNOWN": 1.0})
        assert fig.data[0].marker.color[0] == "#888888"

    def test_x_labels_match_class_keys(self):
        probs = {"NORM": 0.6, "MI": 0.4}
        fig = create_probability_chart(probs)
        assert list(fig.data[0].x) == ["NORM", "MI"]

    def test_yaxis_range(self):
        fig = create_probability_chart({"NORM": 0.9})
        assert list(fig.layout.yaxis.range) == [0, 100]

    @pytest.mark.parametrize(("class_name", "expected_color"), list(CLASS_COLORS.items()))
    def test_all_class_colors(self, class_name, expected_color):
        fig = create_probability_chart({class_name: 1.0})
        assert fig.data[0].marker.color[0] == expected_color


class TestCreatePatientExamTimeline:
    @staticmethod
    def _make_history(n: int, cls: str = "NORM") -> list[dict]:
        return [
            {
                "exam_id": f"e{i}",
                "diagnosis_class": cls,
                "timestamp_processed": f"2026-01-0{i + 1}T00:00:00",
            }
            for i in range(n)
        ]

    def test_returns_figure(self):
        fig = create_patient_exam_timeline(self._make_history(2), patient_id=1)
        assert isinstance(fig, go.Figure)

    def test_empty_history_returns_empty_figure(self):
        fig = create_patient_exam_timeline([], patient_id=1)
        assert len(fig.data) == 0

    def test_single_trace_for_patient(self):
        fig = create_patient_exam_timeline(self._make_history(3), patient_id=42)
        assert len(fig.data) == 1

    def test_x_axis_has_correct_exam_numbers(self):
        fig = create_patient_exam_timeline(self._make_history(3), patient_id=1)
        assert list(fig.data[0].x) == [1, 2, 3]

    def test_y_axis_has_diagnosis_classes(self):
        history = [
            {"exam_id": "e1", "diagnosis_class": "NORM", "timestamp_processed": "t1"},
            {"exam_id": "e2", "diagnosis_class": "MI", "timestamp_processed": "t2"},
        ]
        fig = create_patient_exam_timeline(history, patient_id=1)
        assert list(fig.data[0].y) == ["NORM", "MI"]

    def test_changed_exam_uses_star_symbol(self):
        history = [
            {"exam_id": "e1", "diagnosis_class": "NORM", "timestamp_processed": "t1"},
            {"exam_id": "e2", "diagnosis_class": "MI", "timestamp_processed": "t2"},
        ]
        fig = create_patient_exam_timeline(history, patient_id=1)
        symbols = list(fig.data[0].marker.symbol)
        assert symbols[0] == "circle"
        assert symbols[1] == "star"

    def test_unchanged_exam_uses_circle_symbol(self):
        fig = create_patient_exam_timeline(self._make_history(3, cls="NORM"), patient_id=1)
        symbols = list(fig.data[0].marker.symbol)
        assert all(s == "circle" for s in symbols)

    def test_title_contains_patient_id(self):
        fig = create_patient_exam_timeline(self._make_history(2), patient_id=99)
        assert "99" in fig.layout.title.text
