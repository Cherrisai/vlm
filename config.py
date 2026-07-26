"""
Central configuration for Vision Intelligence Studio.

All runtime settings are resolved here from environment variables with
sane production defaults. No other module should read os.environ directly;
they should import AppConfig from this module instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ModelConfig:
    clip_model_id: str = os.environ.get("VIS_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
    llava_model_id: str = os.environ.get("VIS_LLAVA_MODEL_ID", "vikhyatk/moondream2")
    device_preference: str = os.environ.get("VIS_DEVICE", "auto")
    image_size: int = _get_int("VIS_IMAGE_SIZE", 224)
    max_new_tokens: int = _get_int("VIS_MAX_NEW_TOKENS", 256)
    temperature: float = _get_float("VIS_TEMPERATURE", 0.2)
    top_p: float = _get_float("VIS_TOP_P", 0.9)
    batch_size: int = _get_int("VIS_BATCH_SIZE", 8)
    load_in_4bit: bool = _get_bool("VIS_LOAD_IN_4BIT", False)
    use_fast_processor: bool = True


@dataclass(frozen=True)
class DatabaseConfig:
    db_path: Path = DATA_DIR / "vision_studio.db"


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Vision Intelligence Studio"
    author: str = "Saivignesh"
    copyright_notice: str = "Copyright (c) Saivignesh"
    version: str = "1.0.0"
    log_level: str = os.environ.get("VIS_LOG_LEVEL", "INFO")
    log_dir: Path = BASE_DIR / "logs"
    theme: str = os.environ.get("VIS_THEME", "Light")
    model: ModelConfig = field(default_factory=ModelConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    retrieval_dataset_dir: Path = DATA_DIR / "retrieval_dataset"
    max_upload_size_mb: int = _get_int("VIS_MAX_UPLOAD_MB", 20)
    chars_per_token_estimate: float = 4.0


SETTINGS = AppConfig()
SETTINGS.log_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.retrieval_dataset_dir.mkdir(parents=True, exist_ok=True)
