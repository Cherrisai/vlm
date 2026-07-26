"""Session-scoped effective settings, falling back to the static AppConfig."""

from __future__ import annotations

import streamlit as st

from config import SETTINGS

_DEFAULTS = {
    "settings_clip_model_id": SETTINGS.model.clip_model_id,
    "settings_llava_model_id": SETTINGS.model.llava_model_id,
    "settings_device": SETTINGS.model.device_preference,
    "settings_image_size": SETTINGS.model.image_size,
    "settings_temperature": SETTINGS.model.temperature,
    "settings_top_p": SETTINGS.model.top_p,
    "settings_max_tokens": SETTINGS.model.max_new_tokens,
    "settings_batch_size": SETTINGS.model.batch_size,
    "settings_theme": SETTINGS.theme,
}


def ensure_defaults() -> None:
    """Populate st.session_state with default setting values if not already present."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_effective_clip_model_id() -> str:
    ensure_defaults()
    return st.session_state["settings_clip_model_id"]


def get_effective_llava_model_id() -> str:
    ensure_defaults()
    return st.session_state["settings_llava_model_id"]


def get_effective_device() -> str:
    ensure_defaults()
    return st.session_state["settings_device"]


def get_effective_generation_params() -> dict:
    ensure_defaults()
    return {
        "max_new_tokens": st.session_state["settings_max_tokens"],
        "temperature": st.session_state["settings_temperature"],
        "top_p": st.session_state["settings_top_p"],
    }
