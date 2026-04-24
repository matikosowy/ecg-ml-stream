"""Streamlit monitoring dashboard for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import os
from collections import deque
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ecg_ml_stream.config import cfg
from ecg_ml_stream.dashboard.auxiliary import (
    get_kafka_consumer,
    get_topic_message_count,
    parse_diagnosis_message,
)
from ecg_ml_stream.dashboard.plotting import (
    create_ecg_plot,
    create_patient_exam_timeline,
    create_probability_chart,
)
from ecg_ml_stream.dashboard.trend import PatientHistoryTracker
from ecg_ml_stream.utils.constants import (
    CLASS_COLORS,
    CLASS_DESCRIPTIONS,
    ECG_LEAD_NAMES,
)
from ecg_ml_stream.utils.mappings import CLASS_TRANSLATIONS

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
            "Broker Kafki",
            value=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", cfg.kafka.bootstrap_servers),
        )
        topic = st.text_input("Temat Kafki", value=cfg.kafka.topic_diagnoses)
        auto_refresh = st.checkbox("Auto-odświeżanie", value=True)
        refresh_interval = st.slider(
            "Częstotliwość odświeżania (s)", 1, 10, cfg.dashboard.refresh_interval
        )
        max_records = st.slider("Wyświetlane rekordy", 10, 100, cfg.dashboard.max_records)

        st.markdown("---")
        st.header("Filtrowanie")

        show_dangerous_only = st.checkbox("Tylko niebezpieczne")
        selected_classes = st.multiselect(
            "Filtruj po klasach",
            options=list(CLASS_DESCRIPTIONS.keys()),
            default=list(CLASS_DESCRIPTIONS.keys()),
        )

    # Session state init
    if "diagnoses" not in st.session_state:
        st.session_state.diagnoses = deque(maxlen=cfg.dashboard.max_stored)

    if "total_exams" not in st.session_state:
        st.session_state.total_exams = 0

    if "last_update" not in st.session_state:
        st.session_state.last_update = datetime.now()  # noqa: DTZ005 - No timezone needed

    if "selected_exam" not in st.session_state:
        st.session_state.selected_exam = None

    if "patient_tracker" not in st.session_state:
        st.session_state.patient_tracker = PatientHistoryTracker()

    # Pull new messages from Kafka
    if auto_refresh:
        try:
            topic_count = get_topic_message_count(kafka_servers, topic)
            if topic_count is not None:
                st.session_state.total_exams = topic_count

            consumer = get_kafka_consumer(
                bootstrap_servers=kafka_servers,
                topic=topic,
                group_id=cfg.kafka.consumer_group,
            )
            if consumer:
                count = 0
                try:
                    for message in consumer:
                        parsed = parse_diagnosis_message(message.value)
                        if parsed:
                            comparison = st.session_state.patient_tracker.update(parsed)
                            parsed.update(comparison)
                            st.session_state.diagnoses.appendleft(parsed)
                            count += 1
                finally:
                    consumer.close()
                if count > 0:
                    st.session_state.last_update = datetime.now()  # noqa: DTZ005
        except Exception as e:  # noqa: BLE001 - Catch all exceptions for connection errors
            st.warning(f"Cannot connect to Kafka: {e}")

    # Top row: metrics
    col1, col2, col3, col4 = st.columns(4)
    all_diagnoses = list(st.session_state.diagnoses)
    normal_count = sum(1 for d in all_diagnoses if d.get("diagnosis_class") == "NORM")
    dangerous_count = sum(1 for d in all_diagnoses if d.get("is_dangerous"))

    with col1:
        st.metric("Wszystkie badania", st.session_state.total_exams)
    with col2:
        st.metric("W normie", normal_count)
    with col3:
        st.metric("Wymaga interwencji", dangerous_count)
    with col4:
        avg_time = (
            np.mean([d.get("processing_time_ms", 0) for d in all_diagnoses]) if all_diagnoses else 0
        )
        st.metric("Śr. czas przetwarzania (ms)", f"{avg_time:.0f}")

    st.markdown("---")

    left_col, right_col = st.columns([2, 3])

    with left_col:
        st.subheader("Lista badań")

        filtered_diagnoses = [
            d
            for d in all_diagnoses
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
                        exam_num = diag.get("exam_number")
                        class_changed = diag.get("class_changed", False)
                        prev_class = diag.get("prev_diagnosis_class")

                        if exam_num and class_changed and prev_class:
                            exam_label = f" :orange[(badanie nr {exam_num}, {prev_class} → {diag['diagnosis_class']})]"
                        elif exam_num:
                            exam_label = f" (badanie nr {exam_num})"
                        else:
                            exam_label = ""

                        st.markdown(
                            f"**{icon} {diag['diagnosis_class']}**"
                            f" - {diag['diagnosis_probability'] * 100:.1f}%{exam_label}  \n"
                            f"{diag['hospital_name']}  \n"
                            f"{ts}"
                        )

                    with col_b:
                        if st.button("Szczegóły", key=f"btn_{idx}_{diag['exam_id'][:8]}"):
                            st.session_state.selected_exam = diag

                    st.markdown("---")

    with right_col:
        detail_header, detail_close = st.columns([5, 1])
        with detail_header:
            st.subheader("Szczegóły badania")
        with detail_close:
            if st.session_state.selected_exam is not None:
                if st.button("✕ Zamknij", key="close_detail"):
                    st.session_state.selected_exam = None
                    st.rerun()

        selected = st.session_state.selected_exam

        if selected:
            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.markdown(
                    f"**ID:** `{selected['exam_id'][:16]}...`  \n"
                    f"**Szpital:** {selected['hospital_name']}  \n"
                    f"**Miasto:** {selected['hospital_city']}  \n"
                    f"**Czas przetwarzania:** {selected['processing_time_ms']:.0f} ms"
                )

            with col_info2:
                patient_id = selected.get("patient_id")
                exam_number = selected.get("exam_number")
                if patient_id is not None and exam_number is not None:
                    patient_label = f"Pacjent ID {patient_id} — badanie nr {exam_number}"
                else:
                    patient_label = "Brak danych o pacjencie"

                st.markdown(
                    f"**Pacjent:**  \n"
                    f"- {patient_label}  \n"
                    f"- Wiek: {selected['patient_age'] or 'brak'}  \n"
                    f"- Płeć: {selected['patient_sex'] or 'brak'}  \n"
                    f"- ID EKG: {selected['patient_ecg_id']}"
                )

            st.markdown("---")

            is_dangerous = selected["is_dangerous"]
            diag_class = selected["diagnosis_class"]
            diag_prob = selected["diagnosis_probability"] * 100

            desc = CLASS_DESCRIPTIONS.get(diag_class, "")
            translated = CLASS_TRANSLATIONS.get(desc, desc)
            diagnosis_text = f"**Diagnoza: {diag_class}** ({diag_prob:.1f}%). \n{translated}"

            if is_dangerous:
                st.error(diagnosis_text)
            else:
                st.success(diagnosis_text)

            # Inter-exam comparison (shown only for 2nd+ exam of a patient)
            exam_number = selected.get("exam_number")
            prev_class = selected.get("prev_diagnosis_class")
            if exam_number is not None and exam_number > 1 and prev_class is not None:
                class_changed = selected.get("class_changed", False)
                feature_deviation = selected.get("feature_deviation")

                if class_changed:
                    st.warning(
                        f"Zmiana diagnozy: **{prev_class}** → **{diag_class}**"
                    )
                else:
                    st.info(f"Diagnoza bez zmian od poprzedniego badania: **{prev_class}**")

                if feature_deviation is not None:
                    st.markdown(
                        f"**Odchylenie cech sygnału od poprzedniego badania:** "
                        f"{feature_deviation:.4f}"
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
                    default=cfg.dashboard.default_leads,
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
                d.get("processing_time_ms", 0) for d in all_diagnoses if d.get("processing_time_ms")
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
                    1 for d in all_diagnoses if d.get("ground_truth") == d.get("diagnosis_class")
                )
                total_with_gt = sum(1 for d in all_diagnoses if d.get("ground_truth"))
                if total_with_gt > 0:
                    accuracy = correct / total_with_gt * 100
                    st.markdown(f"- **{accuracy:.1f}%** ({correct}/{total_with_gt})")

    # Historia pacjentów
    st.markdown("---")
    st.subheader("Historia pacjentów")

    tracker = st.session_state.patient_tracker
    patient_stats = tracker.get_stats()

    col_ps1, col_ps2 = st.columns(2)
    with col_ps1:
        st.metric("Unikalnych pacjentów", patient_stats["total_patients"])
    with col_ps2:
        st.metric("Pacjenci z powtórnymi badaniami", patient_stats["patients_with_history"])

    selected = st.session_state.selected_exam
    if selected:
        patient_id = selected.get("patient_id")
        if patient_id is not None:
            patient_history = tracker.get_patient_history(patient_id)
            if len(patient_history) > 1:
                st.plotly_chart(
                    create_patient_exam_timeline(patient_history, patient_id),
                    use_container_width=True,
                )
            elif len(patient_history) == 1:
                st.info(
                    f"Pacjent {patient_id} ma tylko jedno badanie — "
                    "historia pojawi się po kolejnym badaniu."
                )
    elif patient_stats["total_patients"] == 0:
        st.info("Brak danych o historii pacjentów...")
    else:
        st.info("Wybierz badanie z listy, aby zobaczyć historię pacjenta.")

    if auto_refresh and st.session_state.selected_exam is None:
        st_autorefresh(interval=refresh_interval * 1000, key="ecg_autorefresh")


if __name__ == "__main__":
    main()
