"""Numerical and system metric helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import psutil
import torch


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy vectors."""
    a = vector_a.flatten().astype(np.float64)
    b = vector_b.flatten().astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)


def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute Euclidean distance between two 1-D numpy vectors."""
    a = vector_a.flatten().astype(np.float64)
    b = vector_b.flatten().astype(np.float64)
    return float(np.linalg.norm(a - b))


def similarity_to_confidence(similarity: float) -> float:
    """Map a cosine similarity in [-1, 1] to a 0-100 confidence percentage."""
    clipped = max(-1.0, min(1.0, similarity))
    return round(((clipped + 1.0) / 2.0) * 100.0, 2)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D numpy array."""
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the number of tokens for a piece of text using a character heuristic."""
    if not text:
        return 0
    return max(1, int(round(len(text) / chars_per_token)))


@dataclass
class TimedResult:
    elapsed_seconds: float


@contextmanager
def timer():
    """Context manager that yields a TimedResult populated on exit."""
    result = TimedResult(elapsed_seconds=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = round(time.perf_counter() - start, 4)


def get_system_status() -> dict:
    """Return current CPU, memory, and GPU status information."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    virtual_memory = psutil.virtual_memory()
    status = {
        "cpu_percent": cpu_percent,
        "memory_percent": virtual_memory.percent,
        "memory_used_gb": round(virtual_memory.used / (1024 ** 3), 2),
        "memory_total_gb": round(virtual_memory.total / (1024 ** 3), 2),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        status["gpu_name"] = torch.cuda.get_device_name(0)
        status["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        status["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)
    else:
        status["gpu_name"] = "N/A"
        status["gpu_memory_allocated_gb"] = 0.0
        status["gpu_memory_reserved_gb"] = 0.0
    return status
