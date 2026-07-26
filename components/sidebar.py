"""Sidebar navigation component."""

from __future__ import annotations

import streamlit as st

from config import SETTINGS
from utils.constants import NAV_ITEMS


def render_sidebar() -> str:
    """Render the sidebar navigation and return the currently selected page."""
    with st.sidebar:
        st.markdown(f"### {SETTINGS.app_name}")
        st.caption(f"Version {SETTINGS.version}")
        st.divider()

        selected = st.radio(
            label="Navigation",
            options=NAV_ITEMS,
            label_visibility="collapsed",
            key="nav_selection",
        )

        st.divider()
        st.caption(SETTINGS.copyright_notice)

    return selected
