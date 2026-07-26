"""Prompt history and analytics service backed by SQLite."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from database.sqlite import SQLiteManager
from utils.helpers import current_timestamp
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HistoryEntry:
    task_type: str
    image_name: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    inference_time: float


@dataclass
class RequestLogEntry:
    request_type: str
    task_type: str
    similarity_score: float | None
    inference_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int


class HistoryService:
    """Persists and retrieves prompt history and request analytics."""

    def __init__(self, db: SQLiteManager) -> None:
        self.db = db

    def log_prompt(self, entry: HistoryEntry) -> int:
        """Insert a prompt/response record into prompt_history."""
        query = """
            INSERT INTO prompt_history
            (task_type, image_name, prompt, response, input_tokens, output_tokens,
             total_tokens, inference_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entry.task_type,
            entry.image_name,
            entry.prompt,
            entry.response,
            entry.input_tokens,
            entry.output_tokens,
            entry.total_tokens,
            entry.inference_time,
            current_timestamp(),
        )
        return self.db.execute(query, params)

    def log_request(self, entry: RequestLogEntry) -> int:
        """Insert a request record into request_log for analytics purposes."""
        query = """
            INSERT INTO request_log
            (request_type, task_type, similarity_score, inference_time,
             input_tokens, output_tokens, total_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entry.request_type,
            entry.task_type,
            entry.similarity_score,
            entry.inference_time,
            entry.input_tokens,
            entry.output_tokens,
            entry.total_tokens,
            current_timestamp(),
        )
        return self.db.execute(query, params)

    def get_history_dataframe(
        self, search_term: str | None = None, task_filter: str | None = None
    ) -> pd.DataFrame:
        """Retrieve prompt history as a pandas DataFrame, optionally filtered."""
        query = "SELECT * FROM prompt_history WHERE 1=1"
        params: list = []

        if task_filter and task_filter != "All":
            query += " AND task_type = ?"
            params.append(task_filter)

        if search_term:
            query += " AND (prompt LIKE ? OR response LIKE ? OR image_name LIKE ?)"
            like_term = f"%{search_term}%"
            params.extend([like_term, like_term, like_term])

        query += " ORDER BY created_at DESC"
        rows = self.db.fetch_all(query, tuple(params))
        return pd.DataFrame([dict(row) for row in rows])

    def delete_entry(self, entry_id: int) -> None:
        """Delete a single prompt history entry by id."""
        self.db.execute("DELETE FROM prompt_history WHERE id = ?", (entry_id,))

    def clear_history(self) -> None:
        """Delete all prompt history entries."""
        self.db.execute("DELETE FROM prompt_history")

    def get_request_log_dataframe(self) -> pd.DataFrame:
        """Retrieve the full request log as a pandas DataFrame."""
        rows = self.db.fetch_all("SELECT * FROM request_log ORDER BY created_at DESC")
        return pd.DataFrame([dict(row) for row in rows])

    def get_session_summary(self) -> dict:
        """Compute aggregate analytics across all logged requests."""
        df = self.get_request_log_dataframe()
        if df.empty:
            return {
                "total_requests": 0,
                "clip_requests": 0,
                "llava_requests": 0,
                "average_response_time": 0.0,
                "average_similarity": 0.0,
                "total_tokens": 0,
            }

        clip_requests = int((df["request_type"] == "clip").sum())
        llava_requests = int((df["request_type"] == "llava").sum())
        similarity_values = df["similarity_score"].dropna()

        return {
            "total_requests": int(len(df)),
            "clip_requests": clip_requests,
            "llava_requests": llava_requests,
            "average_response_time": round(float(df["inference_time"].mean()), 4),
            "average_similarity": round(float(similarity_values.mean()), 4)
            if not similarity_values.empty
            else 0.0,
            "total_tokens": int(df["total_tokens"].sum()),
        }
