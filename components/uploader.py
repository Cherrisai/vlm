"""Image uploader component with validation."""

from __future__ import annotations

from PIL import Image

import streamlit as st

from config import SETTINGS
from utils.constants import SUPPORTED_IMAGE_TYPES
from utils.image_utils import load_image_from_bytes


def render_image_uploader(label: str, key: str) -> tuple[Image.Image | None, str | None]:
    """Render a single image uploader and return the loaded image and its file name."""
    uploaded_file = st.file_uploader(
        label=label,
        type=SUPPORTED_IMAGE_TYPES,
        key=key,
    )

    if uploaded_file is None:
        return None, None

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > SETTINGS.max_upload_size_mb:
        st.error(f"File exceeds the maximum allowed size of {SETTINGS.max_upload_size_mb} MB.")
        return None, None

    try:
        image = load_image_from_bytes(uploaded_file.getvalue())
    except Exception:
        st.error("The uploaded file could not be read as a valid image.")
        return None, None

    return image, uploaded_file.name


def render_multi_image_uploader(label: str, key: str) -> list[tuple[Image.Image, str]]:
    """Render a multi-file image uploader and return a list of (image, name) pairs."""
    uploaded_files = st.file_uploader(
        label=label,
        type=SUPPORTED_IMAGE_TYPES,
        key=key,
        accept_multiple_files=True,
    )

    results: list[tuple[Image.Image, str]] = []
    if not uploaded_files:
        return results

    for uploaded_file in uploaded_files:
        try:
            image = load_image_from_bytes(uploaded_file.getvalue())
            results.append((image, uploaded_file.name))
        except Exception:
            st.warning(f"Skipped unreadable file: {uploaded_file.name}")

    return results
