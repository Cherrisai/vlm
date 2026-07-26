"""Dashboard landing page."""

from __future__ import annotations

import streamlit as st

from components.cards import render_metric_row, render_section_header
from components.charts import render_pie_chart
from services.history_service import HistoryService
from utils.metrics import get_system_status


def render(history_service: HistoryService) -> None:
    """Render the dashboard overview page."""
    render_section_header(
        "Vision Intelligence Studio",
        "A unified workspace for CLIP and LLaVA powered vision-language intelligence.",
    )

    summary = history_service.get_session_summary()
    render_metric_row(
        [
            ("Total Requests", str(summary["total_requests"]), None),
            ("CLIP Requests", str(summary["clip_requests"]), None),
            ("LLaVA Requests", str(summary["llava_requests"]), None),
            ("Avg Response Time", f"{summary['average_response_time']} s", None),
        ]
    )

    st.write("")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Request Breakdown")
        if summary["total_requests"] > 0:
            render_pie_chart(
                labels=["CLIP", "LLaVA"],
                values=[summary["clip_requests"], summary["llava_requests"]],
                title="Requests by Model Type",
            )
        else:
            st.info("No requests logged yet. Try a feature from the sidebar.")

    with col_right:
        st.markdown("### System Status")
        status = get_system_status()
        render_metric_row(
            [
                ("CPU Usage", f"{status['cpu_percent']}%", None),
                ("Memory Usage", f"{status['memory_percent']}%", None),
                ("CUDA Available", "Yes" if status["cuda_available"] else "No", None),
            ]
        )
        st.caption(f"GPU: {status['gpu_name']}")

    st.write("")
    st.markdown("### Feature Overview")
    features = [
        "Image-Image Similarity using CLIP embeddings",
        "Image-Text Matching and ranking with CLIP",
        "Top-K Image Retrieval from a local dataset",
        "Detailed Image Captioning powered by LLaVA",
        "Multi-turn Visual Question Answering",
        "Structured Image Analysis Dashboard",
        "OCR Assistance: explain, summarize, translate",
        "Prompt History with CSV export",
        "Session Analytics and Token Dashboard",
        "Performance Monitor for inference and system resources",
    ]
    for feature in features:
        st.markdown(f"- {feature}")
