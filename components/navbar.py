"""Top navigation bar component showing the current page title and device."""

from __future__ import annotations

import streamlit as st

from models.loader import resolve_device


def render_navbar(page_title: str) -> None:
    """Render a top bar with the current page title and active compute device."""
    device = resolve_device()

    col_title, col_device = st.columns([4, 1])
    with col_title:
        st.markdown(f"#### {page_title}")
    with col_device:
        st.caption("Device")
        st.markdown(f"**{device.upper()}**")

    st.divider()
