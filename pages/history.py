"""Prompt history browsing, search, filter, export, and delete page."""

from __future__ import annotations

from components.cards import render_section_header
from components.history_table import render_history_controls, render_history_table
from services.history_service import HistoryService


def render(history_service: HistoryService) -> None:
    """Render the prompt history page."""
    render_section_header("Prompt History", "Browse, search, filter, export, and manage past requests.")

    search_term, task_filter = render_history_controls()
    df = history_service.get_history_dataframe(search_term=search_term, task_filter=task_filter)
    render_history_table(df, history_service)
