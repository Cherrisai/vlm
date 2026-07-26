"""General purpose helper functions."""

from __future__ import annotations

from datetime import datetime


def current_timestamp() -> str:
    """Return the current timestamp formatted as an ISO-like string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to a maximum length, appending an ellipsis when shortened."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning a default value when the denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def format_bytes(num_bytes: float) -> str:
    """Format a byte count into a human readable string."""
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value within an inclusive range."""
    return max(low, min(high, value))


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into consecutive chunks of a given size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
