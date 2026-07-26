"""Settings page: model selection, device, generation parameters, theme, and cache control."""

from __future__ import annotations

import streamlit as st

from components.cards import render_section_header
from config import SETTINGS
from utils.session_settings import ensure_defaults

CLIP_MODEL_OPTIONS = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
]

LLAVA_MODEL_OPTIONS = [
    "vikhyatk/moondream2",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/bakLlava-v1-hf",
]

DEVICE_OPTIONS = ["auto", "cpu", "cuda"]
THEME_OPTIONS = ["Light", "Dark"]


def render() -> None:
    """Render the settings page and persist choices into session state."""
    render_section_header("Settings", "Configure model selection, device, and generation parameters.")

    ensure_defaults()

    with st.form("settings_form"):
        st.markdown("#### Model Selection")
        col_clip, col_llava = st.columns(2)
        with col_clip:
            clip_model_id = st.selectbox(
                "CLIP Model",
                options=CLIP_MODEL_OPTIONS,
                index=CLIP_MODEL_OPTIONS.index(st.session_state["settings_clip_model_id"])
                if st.session_state["settings_clip_model_id"] in CLIP_MODEL_OPTIONS
                else 0,
            )
        with col_llava:
            llava_model_id = st.selectbox(
                "LLaVA Model",
                options=LLAVA_MODEL_OPTIONS,
                index=LLAVA_MODEL_OPTIONS.index(st.session_state["settings_llava_model_id"])
                if st.session_state["settings_llava_model_id"] in LLAVA_MODEL_OPTIONS
                else 0,
            )

        st.markdown("#### Device")
        device = st.selectbox(
            "Compute Device",
            options=DEVICE_OPTIONS,
            index=DEVICE_OPTIONS.index(st.session_state["settings_device"])
            if st.session_state["settings_device"] in DEVICE_OPTIONS
            else 0,
        )

        st.markdown("#### Generation Parameters")
        col_a, col_b = st.columns(2)
        with col_a:
            image_size = st.slider(
                "Image Size", min_value=128, max_value=512, step=32,
                value=int(st.session_state["settings_image_size"]),
            )
            temperature = st.slider(
                "Temperature", min_value=0.0, max_value=1.5, step=0.05,
                value=float(st.session_state["settings_temperature"]),
            )
            top_p = st.slider(
                "Top P", min_value=0.1, max_value=1.0, step=0.05,
                value=float(st.session_state["settings_top_p"]),
            )
        with col_b:
            max_tokens = st.slider(
                "Max Tokens", min_value=32, max_value=1024, step=32,
                value=int(st.session_state["settings_max_tokens"]),
            )
            batch_size = st.slider(
                "Batch Size", min_value=1, max_value=32, step=1,
                value=int(st.session_state["settings_batch_size"]),
            )
            theme = st.selectbox(
                "Theme",
                options=THEME_OPTIONS,
                index=THEME_OPTIONS.index(st.session_state["settings_theme"])
                if st.session_state["settings_theme"] in THEME_OPTIONS
                else 0,
            )

        submitted = st.form_submit_button("Save Settings", type="primary")

        if submitted:
            st.session_state["settings_clip_model_id"] = clip_model_id
            st.session_state["settings_llava_model_id"] = llava_model_id
            st.session_state["settings_device"] = device
            st.session_state["settings_image_size"] = image_size
            st.session_state["settings_temperature"] = temperature
            st.session_state["settings_top_p"] = top_p
            st.session_state["settings_max_tokens"] = max_tokens
            st.session_state["settings_batch_size"] = batch_size
            st.session_state["settings_theme"] = theme
            st.success(
                "Settings saved. Model changes take effect the next time a model is loaded."
            )

    st.divider()
    st.markdown("#### Cache Management")
    st.caption(
        "Clearing the cache forces CLIP and LLaVA models to be reloaded from disk or the "
        "Hugging Face Hub on their next use."
    )
    if st.button("Clear Model Cache", type="secondary"):
        st.cache_resource.clear()
        st.success("Model cache cleared successfully.")

    st.divider()
    st.markdown("#### Current Effective Configuration")
    st.json(
        {
            "clip_model_id": st.session_state["settings_clip_model_id"],
            "llava_model_id": st.session_state["settings_llava_model_id"],
            "device": st.session_state["settings_device"],
            "image_size": st.session_state["settings_image_size"],
            "temperature": st.session_state["settings_temperature"],
            "top_p": st.session_state["settings_top_p"],
            "max_tokens": st.session_state["settings_max_tokens"],
            "batch_size": st.session_state["settings_batch_size"],
            "theme": st.session_state["settings_theme"],
            "database_path": str(SETTINGS.database.db_path),
        }
    )
