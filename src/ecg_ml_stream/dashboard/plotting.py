"""Plotting module for Streamlit dashboard app for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ecg_ml_stream.utils.constants import CLASS_COLORS, ECG_LEAD_NAMES, NUM_LEADS


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


def create_centroid_chart(
    centroids: dict[str, np.ndarray],
    n_leads: int = NUM_LEADS,
) -> go.Figure:
    """Render grouped bar chart of per-class mean signal amplitude per lead.

    Args:
        centroids: Dict mapping class name to centroid array of shape
            (n_leads * 2,) where even indices are mean amplitudes and odd
            indices are standard deviations.
        n_leads: Number of ECG leads.

    Returns:
        go.Figure: Plotly Figure with one bar group per class.

    """
    fig = go.Figure()
    lead_labels = ECG_LEAD_NAMES[:n_leads]

    for cls, centroid in centroids.items():
        amplitudes = [centroid[i * 2] for i in range(n_leads)]
        fig.add_trace(
            go.Bar(
                name=cls,
                x=lead_labels,
                y=amplitudes,
                marker_color=CLASS_COLORS.get(cls, "#888888"),
                text=[f"{v:.3f}" for v in amplitudes],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Średnia amplituda sygnalu per sonda i klasa",
        xaxis_title="Sonda EKG",
        yaxis_title="Średnia amplituda",
        barmode="group",
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def create_deviation_timeline(
    history: list[dict],
    class_names: list[str],
) -> go.Figure:
    """Render a scatter plot of per-sample deviation scores over time.

    Args:
        history: List of dicts with timestamp, deviation, exam_id, class_name.
        class_names: Ordered list of class names for trace grouping.

    Returns:
        go.Figure: Plotly Figure with one scatter trace per class.

    """
    fig = go.Figure()

    grouped: dict[str, list[dict]] = {c: [] for c in class_names}
    for entry in history:
        cls = entry.get("class_name", "")
        if cls in grouped:
            grouped[cls].append(entry)

    for cls in class_names:
        entries = grouped[cls]
        if not entries:
            continue

        fig.add_trace(
            go.Scatter(
                x=[e["timestamp"] for e in entries],
                y=[e["deviation"] for e in entries],
                mode="markers",
                name=cls,
                marker={
                    "color": CLASS_COLORS.get(cls, "#888888"),
                    "size": 6,
                },
                hovertemplate=(
                    "%{text}<br>"
                    "Odchylenie: %{y:.4f}<br>"
                    "<extra>%{fullData.name}</extra>"
                ),
                text=[e.get("exam_id", "")[:8] for e in entries],
            )
        )

    fig.update_layout(
        title="Odchylenie od centroidu klasy w czasie",
        xaxis_title="Czas",
        yaxis_title="Odchylenie (odległość euklidesowa)",
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig
