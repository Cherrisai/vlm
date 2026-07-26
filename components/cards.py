"""Reusable card and metric display components."""

from __future__ import annotations

import streamlit as st


def render_metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a row of st.metric widgets from (label, value, delta) tuples."""
    columns = st.columns(len(metrics))
    for column, (label, value, delta) in zip(columns, metrics):
        with column:
            st.metric(label=label, value=value, delta=delta)


def render_result_card(title: str, content: str, footer: str | None = None) -> None:
    """Render a bordered result card with a title, body content, and optional footer."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(content)
        if footer:
            st.caption(footer)


def render_status_badge(label: str, status: str, positive: bool = True) -> None:
    """Render a colored status indicator."""
    if positive:
        st.success(f"{label}: {status}")
    else:
        st.warning(f"{label}: {status}")


def render_section_header(title: str, subtitle: str | None = None) -> None:
    """Render a consistent page section header."""
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()
