"""Analytics page: session analytics, token dashboard, and performance monitor."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import render_metric_row, render_section_header
from components.charts import render_distribution_chart, render_pie_chart, render_timeline_chart
from services.history_service import HistoryService
from utils.metrics import get_system_status


def render(history_service: HistoryService) -> None:
    """Render the analytics page with three tabs covering the required dashboards."""
    render_section_header("Analytics", "Session analytics, token usage, and performance monitoring.")

    tab_session, tab_tokens, tab_performance = st.tabs(
        ["Session Analytics", "Token Dashboard", "Performance Monitor"]
    )

    request_df = history_service.get_request_log_dataframe()
    prompt_df = history_service.get_history_dataframe()

    with tab_session:
        summary = history_service.get_session_summary()
        render_metric_row(
            [
                ("Total Requests", str(summary["total_requests"]), None),
                ("CLIP Requests", str(summary["clip_requests"]), None),
                ("LLaVA Requests", str(summary["llava_requests"]), None),
                ("Avg Response Time", f"{summary['average_response_time']} s", None),
            ]
        )
        render_metric_row(
            [
                ("Average Similarity", f"{summary['average_similarity']}", None),
                ("Total Tokens", str(summary["total_tokens"]), None),
                ("Total Uploaded Images", str(len(prompt_df)) if not prompt_df.empty else "0", None),
            ]
        )

        col_left, col_right = st.columns(2)
        with col_left:
            if not request_df.empty:
                render_pie_chart(
                    labels=["CLIP", "LLaVA"],
                    values=[summary["clip_requests"], summary["llava_requests"]],
                    title="Requests by Model Type",
                )
            else:
                st.info("No requests logged yet.")

        with col_right:
            if not request_df.empty:
                request_df["created_at"] = pd.to_datetime(request_df["created_at"])
                timeline = (
                    request_df.groupby(request_df["created_at"].dt.floor("min"))
                    .size()
                    .reset_index(name="requests")
                )
                render_timeline_chart(timeline, "created_at", "requests", "Request Timeline")
            else:
                st.info("No timeline data available yet.")

        st.markdown("#### Most Asked Questions")
        if not prompt_df.empty:
            question_counts = (
                prompt_df[prompt_df["task_type"] == "visual_qa"]["prompt"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            question_counts.columns = ["Question", "Count"]
            st.dataframe(question_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No visual QA questions logged yet.")

        st.markdown("#### Prompt Length and Similarity Distribution")
        col_len, col_sim = st.columns(2)
        with col_len:
            if not prompt_df.empty:
                lengths = prompt_df["prompt"].astype(str).str.len().tolist()
                render_distribution_chart(lengths, "Prompt Length Distribution", "Characters")
            else:
                st.info("No prompt data available yet.")
        with col_sim:
            if not request_df.empty and request_df["similarity_score"].notna().any():
                similarities = request_df["similarity_score"].dropna().tolist()
                render_distribution_chart(similarities, "Similarity Distribution", "Cosine Similarity")
            else:
                st.info("No similarity data available yet.")

    with tab_tokens:
        if request_df.empty:
            st.info("No token usage recorded yet.")
        else:
            total_input = int(request_df["input_tokens"].sum())
            total_output = int(request_df["output_tokens"].sum())
            total_tokens = int(request_df["total_tokens"].sum())
            conversation_tokens = int(
                request_df.loc[request_df["task_type"] == "visual_qa", "total_tokens"].sum()
            )

            render_metric_row(
                [
                    ("Estimated Input Tokens", str(total_input), None),
                    ("Estimated Output Tokens", str(total_output), None),
                    ("Total Tokens", str(total_tokens), None),
                    ("Conversation Tokens", str(conversation_tokens), None),
                ]
            )

            request_df["created_at"] = pd.to_datetime(request_df["created_at"])
            token_timeline = request_df[["created_at", "total_tokens"]].sort_values("created_at")
            render_timeline_chart(token_timeline, "created_at", "total_tokens", "Token Usage History")

            st.markdown("#### Token Usage by Task")
            by_task = (
                request_df.groupby("task_type")[["input_tokens", "output_tokens", "total_tokens"]]
                .sum()
                .reset_index()
            )
            st.dataframe(by_task, use_container_width=True, hide_index=True)

    with tab_performance:
        status = get_system_status()
        render_metric_row(
            [
                ("CPU Usage", f"{status['cpu_percent']}%", None),
                ("Memory Usage", f"{status['memory_percent']}%", None),
                (
                    "Memory Used",
                    f"{status['memory_used_gb']} / {status['memory_total_gb']} GB",
                    None,
                ),
                ("CUDA Available", "Yes" if status["cuda_available"] else "No", None),
            ]
        )
        render_metric_row(
            [
                ("GPU", status["gpu_name"], None),
                ("GPU Memory Allocated", f"{status['gpu_memory_allocated_gb']} GB", None),
                ("GPU Memory Reserved", f"{status['gpu_memory_reserved_gb']} GB", None),
            ]
        )

        if not request_df.empty:
            st.markdown("#### Inference Time by Task")
            timing = (
                request_df.groupby("task_type")["inference_time"]
                .mean()
                .round(4)
                .reset_index()
                .rename(columns={"inference_time": "avg_inference_time_seconds"})
            )
            st.dataframe(timing, use_container_width=True, hide_index=True)

            request_df["created_at"] = pd.to_datetime(request_df["created_at"])
            timing_series = request_df[["created_at", "inference_time"]].sort_values("created_at")
            render_timeline_chart(
                timing_series, "created_at", "inference_time", "Inference Time Over Requests"
            )
        else:
            st.info("No performance data recorded yet.")
