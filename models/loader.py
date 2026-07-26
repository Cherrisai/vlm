"""Device detection and model loading orchestration."""

from __future__ import annotations

import torch

from config import SETTINGS
from utils.constants import DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA
from utils.logger import get_logger

logger = get_logger(__name__)


def resolve_device(preference: str | None = None) -> str:
    """Resolve the compute device to use based on preference and availability."""
    choice = (preference or SETTINGS.model.device_preference or DEVICE_AUTO).lower()

    if choice == DEVICE_CUDA:
        if torch.cuda.is_available():
            return DEVICE_CUDA
        logger.warning("CUDA requested but not available, falling back to CPU.")
        return DEVICE_CPU

    if choice == DEVICE_CPU:
        return DEVICE_CPU

    return DEVICE_CUDA if torch.cuda.is_available() else DEVICE_CPU


def torch_dtype_for_device(device: str) -> torch.dtype:
    """Return the preferred torch dtype for the given device."""
    return torch.float16 if device == DEVICE_CUDA else torch.float32
