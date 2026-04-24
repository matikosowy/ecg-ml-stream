"""Plotting module for Streamlit dashboard app for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ecg_ml_stream.utils.constants import CLASS_COLORS, ECG_LEAD_NAMES


def create_ecg_plot(
    signal_data: list[list[float]],
    sampling_rate: int,
    selected_leads: list[str] | None = None,
) -> go.Figure:
    """Render a multi-lead ECG signal as a stacked Plotly figure.

    Args:
        signal_data (list[list[float]]): Nested list of shape (num_leads, num_samples).
        sampling_rate (int): Sampling rate of the ECG signal in Hz.
        selected_leads (list[str] | None): List of leads to display.
            If None, all leads are displayed.

    Returns:
        go.Figure: A Plotly figure containing the ECG signal plots.

    """
    if selected_leads is None:
        selected_leads = list(range(len(signal_data)))

    num_leads = len(selected_leads)
    num_samples = len(signal_data[0]) if signal_data else 0
    time_axis = np.arange(num_samples) / sampling_rate

    fig = make_subplots(
        rows=num_leads,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[ECG_LEAD_NAMES[i] for i in selected_leads],
    )

    for idx, lead_idx in enumerate(selected_leads):
        signal = signal_data[lead_idx] if lead_idx < len(signal_data) else []

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal,
                mode="lines",
                name=ECG_LEAD_NAMES[lead_idx],
                line={
                    "color": "#1f77b4",
                    "width": 1,
                },
                showlegend=False,
            ),
            row=idx + 1,
            col=1,
        )

        fig.update_yaxes(
            title_text=ECG_LEAD_NAMES[lead_idx],
            row=idx + 1,
            col=1,
            showgrid=True,
            gridcolor="rgba(255,192,203,0.3)",
            zeroline=True,
            zerolinecolor="rgba(255,0,0,0.3)",
        )

        fig.update_xaxes(
            title_text="Time [s]",
            row=num_leads,
            col=1,
            showgrid=True,
            gridcolor="rgba(255,192,203,0.3)",
        )

        fig.update_layout(
            height=max(400, 80 * num_leads),
            title_text="ECG signal - 12 leads",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={
                "l": 60,
                "r": 20,
                "t": 40,
                "b": 40,
            },
        )
    return fig


def create_probability_chart(probabilities: dict[str, float]) -> go.Figure:
    """Render a bar chart of per-class diagnosis probabilities.

    Args:
        probabilities (dict[str, float]): Dict mapping class names to probability values in [0, 1].

    Returns:
        go.Figure: Plotly Figure with one bar per class.

    """
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]
    colors = [CLASS_COLORS.get(c, "#888888") for c in classes]

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=probs,
                marker_color=colors,
                text=[f"{p:.1f}%" for p in probs],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Prawdopodobieństwa zdiagnozowania klas",
        xaxis_title="Klasa",
        yaxis_title="Prawdopodobieństwo [%]",
        yaxis_range=[0, 100],
        height=300,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def create_patient_exam_timeline(
    patient_history: list[dict],
    patient_id: int,
) -> go.Figure:
    """Render a timeline of diagnosis classes for a single patient's exams.

    Args:
        patient_history: List of exam dicts with diagnosis_class and
            timestamp_processed fields (oldest first).
        patient_id: PTB-XL patient identifier used for the chart title.

    Returns:
        go.Figure: Plotly Figure with one marker per exam.

    """
    if not patient_history:
        return go.Figure()

    exam_numbers = list(range(1, len(patient_history) + 1))
    classes = [e.get("diagnosis_class", "") for e in patient_history]
    timestamps = [e.get("timestamp_processed") or "" for e in patient_history]
    colors = [CLASS_COLORS.get(c, "#888888") for c in classes]
    changed = [False] + [classes[i] != classes[i - 1] for i in range(1, len(classes))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=exam_numbers,
            y=classes,
            mode="markers+lines",
            marker={
                "color": colors,
                "size": [14 if ch else 9 for ch in changed],
                "symbol": ["star" if ch else "circle" for ch in changed],
                "line": {"width": 1, "color": "white"},
            },
            line={"color": "lightgray", "width": 1},
            text=[
                f"Badanie {n}<br>{c}<br>{ts[:19] if ts else 'brak czasu'}"
                for n, c, ts in zip(exam_numbers, classes, timestamps, strict=True)
            ],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Historia badań pacjenta {patient_id}",
        xaxis={
            "title": "Nr badania",
            "tickmode": "linear",
            "dtick": 1,
        },
        yaxis_title="Diagnoza",
        height=300,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
    )
    return fig
