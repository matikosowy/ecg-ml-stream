"""Streamlit monitoring dashboard for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import json
import os

from collections import deque
from datetime import datetime

import numpy as np
from ecg_ml_stream.dashboard.auxiliary import get_kafka_consumer, parse_diagnosis_message
from ecg_ml_stream.dashboard.plotting import create_ecg_plot, create_probability_chart
import plotly.graph_objects as go
import streamlit as st

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from streamlit_autorefresh import st_autorefresh

from ecg_ml_stream.utils.constants import CLASS_DESCRIPTIONS, ECG_LEAD_NAMES, CLASS_COLORS


st.set_page_config(
    page_title="Monitoring EKG",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Render the ECG monitoring dashboard."""
    st.markdown(
        '<div class="main-header">Monitoring EKG</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    with st.sidebar:
        st.header("Konfiguracja")

        kafka_servers = st.text_input(
            "Kafka Bootstrap Servers",
            value=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        )
        topic = st.text_input("Kafka topic", value="ecg-diagnoses")
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (s)", 1, 10, 2)
        max_records = st.slider("Max records", 10, 100, 50)

        st.markdown("---")
        st.header("Filtrowanie")

        show_dangerous_only = st.checkbox("Dangerous diagnoses only")
        selected_classes = st.multiselect(
            "Filter classes",
            options=list(CLASS_DESCRIPTIONS.keys()),
            default=list(CLASS_DESCRIPTIONS.keys()),
        )
    
    # Session state init
    if (
        "diagnoses" not in st.session_state
        or st.session_state.diagnoses.maxlen != max_records
    ):
        old = list(st.session_state.get("diagnoses", []))
        st.session_state.diagnoses = deque(old[:max_records], maxlen=max_records)

    if "last_update" not in st.session_state:
        st.session_state.last_update = datetime.now()

    if "selected_exam" not in st.session_state:
        st.session_state.selected_exam = None
    
    # Pull new messages from Kafka
    if auto_refresh:
        try:
            consumer = get_kafka_consumer(
                bootstrap_servers=kafka_servers,
                topic=topic,
                group_id="dashboard-consumer",
            )
            if consumer:
                for message in consumer:
                    parsed = parse_diagnosis_message(message.value)
                    if parsed:
                        st.session_state.diagnoses.appendleft(parsed)
                        st.session_state.last_update = datetime.now()
                consumer.close()
        except Exception as e:
            st.warning(f"Cannot connect to Kafka: {e}")
    
    # Top row: metrics
    col1, col2, col3, col4 = st.columns(4)
    all_diagnoses = list(st.session_state.diagnoses)
    dangerous_count = sum(1 for d in all_diagnoses if d.get("is_dangerous"))
    normal_count = sum(1 for d in all_diagnoses if d.get("diagnosis_class") == "NORM")

    with col1:
        st.metric("Wszystkie badania", len(all_diagnoses))
    with col2:
        st.metric("W normie", normal_count)
    with col3:
        st.metric("Wymaga interwencji", dangerous_count)
    with col4:
        avg_time = (
            np.mean([d.get("processing_time_ms", 0) for d in all_diagnoses])
            if all_diagnoses
            else 0
        )
        st.metric("Śr. czas przetwarzania (ms)", f"{avg_time:.0f}")

    st.markdown("---")

    left_col, right_col = st.columns([2, 3])

    with left_col:
        st.subheader("Lista badań")

        filtered_diagnoses = [
            d for d in all_diagnoses
            if d.get("diagnosis_class") in selected_classes
            and (not show_dangerous_only or d.get("is_dangerous"))
        ]

        if not filtered_diagnoses:
            st.info("Nie znaleziono badań spełniających kryteria.")
        else:
            for idx, diag in enumerate(filtered_diagnoses[:max_records]):
                is_dangerous = diag.get("is_dangerous", False)

                with st.container():
                    col_a, col_b = st.columns([3, 1])

                    with col_a:
                        icon = "❗" if is_dangerous else "✅"
                        processed = diag.get("timestamp_processed")
                        ts = processed[:19] if processed else "brak"
                        st.markdown(
                            f"**{icon} {diag['diagnosis_class']}**"
                            f" - {diag["diagnosis_probability"] * 100:.1f}%  \n"
                            f"{diag['hospital_name']}  \n"
                            f"{ts}"
                        )
                        
                    with col_b:
                        if st.button("Szczegóły", key=f"btn_{idx}_{diag['exam_id'][:8]}"):
                            st.session_state.selected_exam = diag
                    
                    st.markdown("---")
    
    with right_col:
        st.subheader("Szczegóły badania")
        selected = st.session_state.selected_exam

        if selected:
            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.markdown(
                    f"***ID:** `{selected['exam_id'][:16]}...`  \n"
                    f"***Szpital:** {selected['hospital_name']}  \n"
                    f"**Miasto:** {selected['hospital_city']}  \n"
                    f"**Czas przetwarzania:** {selected['processing_time_ms']:.0f} ms"
                )

            with col_info2:
                st.markdown(
                    f"**Pacjent:**  \n"
                    f"- Wiek: {selected['patient_age'] or 'brak'}  \n"
                    f"- Płeć: {selected['patient_sex'] or 'brak'}  \n"
                    f"- ID badania: {selected['patient_ecg_id']}"
                )
            
            st.markdown("---")

            is_dangerous = selected["is_dangerous"]
            diag_class = selected["diagnosis_class"]
            diag_prob = selected["diagnosis_probability"] * 100

            if is_dangerous:
                st.error(
                    f"**Diagnoza: {diag_class}** ({diag_prob:.1f}%). \n"
                    f"{CLASS_DESCRIPTIONS.get(diag_class, '')}"
                )
            else:
                st.success(
                    f"**Diagnoza: {diag_class}** ({diag_prob:.1f}%). \n"
                    f"{CLASS_DESCRIPTIONS.get(diag_class, '')}"
                )
            
            if selected.get("ground_truth"):
                correct = selected["ground_truth"] == diag_class
                icon = "✅" if correct else "❌"
                st.info(f"Rzeczywista diagnoza: **{selected['ground_truth']}** [{icon}]")
            
            if selected.get("all_probabilities"):
                st.plotly_chart(
                    create_probability_chart(selected["all_probabilities"]),
                    use_container_width=True,
                )
            
            if selected.get("signal_data"):
                st.markdown("### Sygnał EKG")

                lead_options = st.multiselect(
                    "Wybierz sondy",
                    options=list(range(12)),
                    default=[0, 1, 6, 7, 8],
                    format_func=lambda x: ECG_LEAD_NAMES[x],
                )

                if lead_options:
                    st.plotly_chart(
                        create_ecg_plot(
                            selected["signal_data"],
                            selected.get("sampling_rate", 100),
                            lead_options,
                        ),
                        use_container_width=True,
                    )
        else:
            st.info("Wybierz badanie z listy, aby zobaczyć szczegóły.")
    
    # Statystyki
    st.markdown("---")
    st.subheader("Statystyki")

    if all_diagnoses:
        diag_counts = {}
        for d in all_diagnoses:
            cls = d.get("diagnosis_class", "brak")
            diag_counts[cls] = diag_counts.get(cls, 0) + 1

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=list(diag_counts.keys()),
                    values=list(diag_counts.values()),
                    marker_colors=[CLASS_COLORS.get(c, "#888888") for c in diag_counts],
                    hole=0.4,
                )
            ]
        )
        fig_pie.update_layout(title="Rozkład diagnoz", height=300)

        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_stat2:
            times = [
                d.get("processing_time_ms", 0)
                for d in all_diagnoses
                if d.get("processing_time_ms")
            ]
            if times:
                st.markdown(
                    f"**Statystyki przetwarzania:**  \n"
                    f"- Średnio: {np.mean(times):.1f} ms  \n"
                    f"- Min: {np.min(times):.1f} ms  \n"
                    f"- Max: {np.max(times):.1f} ms  \n"
                    f"- Mediana: {np.median(times):.1f} ms  \n"
                    f"\n**Dokładność:**"
                )

                correct = sum(
                    1 for d in all_diagnoses
                    if d.get("ground_truth") == d.get("diagnosis_class")
                )
                total_with_gt = sum(1 for d in all_diagnoses if d.get("ground_truth"))
                if total_with_gt > 0:
                    accuracy = correct / total_with_gt * 100
                    st.markdown(f"- **{accuracy:.1f}%** ({correct}/{total_with_gt})")
            
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="ecg_autorefresh")


if __name__ == "__main__":
    main()
