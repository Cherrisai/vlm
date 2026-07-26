"""History table component with filter, search, export, and delete controls."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.history_service import HistoryService


def render_history_controls() -> tuple[str, str]:
    """Render search and task-type filter controls, returning their current values."""
    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search_term = st.text_input("Search prompts, responses, or image names", value="")
    with col_filter:
        task_filter = st.selectbox(
            "Filter by task",
            options=[
                "All",
                "image_similarity",
                "image_text_matching",
                "image_retrieval",
                "image_captioning",
                "visual_qa",
                "image_analysis",
                "ocr_assistance",
            ],
        )
    return search_term, task_filter


def render_history_table(df: pd.DataFrame, history_service: HistoryService) -> None:
    """Render the history dataframe with export and row-level delete controls."""
    if df.empty:
        st.info("No history entries found.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export history as CSV",
        data=csv_bytes,
        file_name="prompt_history.csv",
        mime="text/csv",
    )

    with st.expander("Delete an entry"):
        entry_id = st.number_input("Entry ID to delete", min_value=0, step=1, value=0)
        if st.button("Delete entry", type="secondary"):
            if entry_id > 0:
                history_service.delete_entry(int(entry_id))
                st.success(f"Deleted entry {entry_id}.")
                st.rerun()
            else:
                st.warning("Enter a valid entry ID.")

    with st.expander("Clear all history"):
        st.warning("This action permanently deletes all stored prompt history.")
        if st.button("Clear all history", type="primary"):
            history_service.clear_history()
            st.success("History cleared.")
            st.rerun()
