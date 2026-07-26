"""Plotly chart building components."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_bar_chart(labels: list[str], values: list[float], title: str, y_label: str) -> None:
    """Render a horizontal bar chart, most commonly used for prompt ranking."""
    figure = go.Figure(
        go.Bar(x=values, y=labels, orientation="h", marker=dict(color="#2563EB"))
    )
    figure.update_layout(
        title=title,
        xaxis_title=y_label,
        yaxis=dict(autorange="reversed"),
        height=max(300, 45 * len(labels)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_probability_table(df: pd.DataFrame) -> None:
    """Render a probability/score table."""
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_timeline_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    """Render a line chart for time-series analytics."""
    if df.empty:
        st.info("No data available yet.")
        return
    figure = px.line(df, x=x_col, y=y_col, title=title, markers=True)
    figure.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(figure, use_container_width=True)


def render_distribution_chart(values: list[float], title: str, x_label: str) -> None:
    """Render a histogram distribution chart."""
    if not values:
        st.info("No data available yet.")
        return
    figure = px.histogram(x=values, nbins=20, title=title)
    figure.update_layout(xaxis_title=x_label, yaxis_title="Count", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(figure, use_container_width=True)


def render_pie_chart(labels: list[str], values: list[float], title: str) -> None:
    """Render a pie chart, commonly used for request type breakdowns."""
    figure = go.Figure(go.Pie(labels=labels, values=values, hole=0.45))
    figure.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(figure, use_container_width=True)
