"""Vision Intelligence Studio - main Streamlit application entrypoint."""

from __future__ import annotations

import streamlit as st

import os
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

from components.navbar import render_navbar
from components.sidebar import render_sidebar
from config import SETTINGS
from database.sqlite import get_db
from pages import (
    analytics,
    captioning,
    dashboard,
    history,
    image_analysis,
    image_similarity,
    image_text_matching,
    retrieval,
    settings as settings_page,
    visual_qa,
)
from services.history_service import HistoryService
from services.token_tracker import TokenTracker
from utils.constants import (
    NAV_ABOUT,
    NAV_ANALYTICS,
    NAV_CAPTIONING,
    NAV_DASHBOARD,
    NAV_IMAGE_ANALYSIS,
    NAV_IMAGE_SIMILARITY,
    NAV_IMAGE_TEXT_MATCHING,
    NAV_PROMPT_HISTORY,
    NAV_RETRIEVAL,
    NAV_SETTINGS,
    NAV_VISUAL_QA,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def configure_page() -> None:
    """Configure global Streamlit page settings."""
    st.set_page_config(
        page_title=SETTINGS.app_name,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_about_page() -> None:
    """Render the About page describing the project and author."""
    st.markdown(f"## About {SETTINGS.app_name}")
    st.divider()
    st.write(
        f"{SETTINGS.app_name} is a production-quality Vision Language Model workspace that "
        "combines OpenAI CLIP for embedding-based similarity and retrieval with LLaVA for "
        "open-ended visual reasoning, captioning, question answering, and OCR assistance."
    )
    st.markdown("#### Capabilities")
    st.markdown(
        "- Image-Image Similarity and Image-Text Matching with CLIP\n"
        "- Top-K Image Retrieval from a local dataset\n"
        "- Detailed captioning, structured analysis, and OCR reasoning with LLaVA\n"
        "- Multi-turn Visual Question Answering\n"
        "- Prompt history, session analytics, token dashboard, and performance monitoring\n"
        "- Configurable models, device, and generation parameters"
    )
    st.markdown("#### Technology Stack")
    st.markdown(
        "Python, PyTorch, Hugging Face Transformers, OpenCLIP, Streamlit, Plotly, "
        "SQLite, Pandas, NumPy, scikit-learn."
    )
    st.divider()
    st.markdown(f"**Version:** {SETTINGS.version}")
    st.markdown(f"**Author:** {SETTINGS.author}")
    st.markdown(f"**{SETTINGS.copyright_notice}**")


@st.cache_resource(show_spinner=False)
def get_history_service() -> HistoryService:
    """Return a cached HistoryService instance backed by the SQLite database."""
    return HistoryService(get_db())


def main() -> None:
    """Application entrypoint."""
    configure_page()

    history_service = get_history_service()
    token_tracker = TokenTracker()

    selected_page = render_sidebar()
    render_navbar(selected_page)

    if selected_page == NAV_DASHBOARD:
        dashboard.render(history_service)
    elif selected_page == NAV_IMAGE_SIMILARITY:
        image_similarity.render(history_service, token_tracker)
    elif selected_page == NAV_IMAGE_TEXT_MATCHING:
        image_text_matching.render(history_service, token_tracker)
    elif selected_page == NAV_RETRIEVAL:
        retrieval.render(history_service, token_tracker)
    elif selected_page == NAV_CAPTIONING:
        captioning.render(history_service, token_tracker)
    elif selected_page == NAV_VISUAL_QA:
        visual_qa.render(history_service, token_tracker)
    elif selected_page == NAV_IMAGE_ANALYSIS:
        image_analysis.render(history_service, token_tracker)
    elif selected_page == NAV_PROMPT_HISTORY:
        history.render(history_service)
    elif selected_page == NAV_ANALYTICS:
        analytics.render(history_service)
    elif selected_page == NAV_SETTINGS:
        settings_page.render()
    elif selected_page == NAV_ABOUT:
        render_about_page()
    else:
        logger.warning("Unknown navigation selection: %s", selected_page)
        dashboard.render(history_service)


if __name__ == "__main__":
    main()
