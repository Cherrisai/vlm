"""Image retrieval page powered by CLIP embeddings."""

from __future__ import annotations

import streamlit as st

from components.cards import render_result_card, render_section_header
from components.uploader import render_image_uploader
from config import SETTINGS
from models.clip_model import get_clip_engine
from services.history_service import HistoryEntry, HistoryService, RequestLogEntry
from services.retrieval import RetrievalService
from services.token_tracker import TokenTracker
from utils.constants import REQUEST_TYPE_CLIP, TASK_RETRIEVAL
from utils.image_utils import load_image_from_path, make_thumbnail
from utils.session_settings import get_effective_clip_model_id, get_effective_device


def render(history_service: HistoryService, token_tracker: TokenTracker) -> None:
    """Render the image retrieval page."""
    render_section_header("Image Retrieval")
    
 

    dataset_files = list(SETTINGS.retrieval_dataset_dir.glob("*"))
    st.caption(f"Dataset currently contains {len(dataset_files)} file(s).")

    with st.expander("Add images to the retrieval dataset"):
        new_files = st.file_uploader(
            "Upload images to add to the dataset",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
            key="retrieval_dataset_upload",
        )
        if new_files and st.button("Save to dataset"):
            for file in new_files:
                target_path = SETTINGS.retrieval_dataset_dir / file.name
                with open(target_path, "wb") as handle:
                    handle.write(file.getvalue())
            st.success(f"Added {len(new_files)} image(s) to the dataset.")
            st.rerun()

    query_image, query_name = render_image_uploader("Upload a query image", key="retrieval_query")
    if query_image is not None:
        st.image(query_image, width=280)

    top_k = st.slider("Number of results (Top-K)", min_value=1, max_value=20, value=5)

    if st.button("Retrieve Similar Images", type="primary", disabled=query_image is None):
        clip_engine = get_clip_engine(
            model_id=get_effective_clip_model_id(), device=get_effective_device()
        )
        retrieval_service = RetrievalService(clip_engine)
        result = retrieval_service.retrieve(query_image, top_k=top_k)

        if result.dataset_size == 0:
            st.warning("The retrieval dataset is empty. Add images above before retrieving.")
            return

        render_result_card(
            title="Retrieval Summary",
            content=f"Compared against {result.dataset_size} dataset image(s).",
            footer=f"Inference time: {result.inference_time} s",
        )

        columns = st.columns(min(5, max(1, len(result.matches))))
        for idx, match in enumerate(result.matches):
            column = columns[idx % len(columns)]
            with column:
                thumbnail = make_thumbnail(load_image_from_path(match.image_path))
                st.image(thumbnail, caption=f"#{match.rank} score={match.score}")
                st.caption(match.image_path.name)

        prompt_text = f"query={query_name}, top_k={top_k}"
        response_text = ", ".join(f"{m.image_path.name}:{m.score}" for m in result.matches)
        usage = token_tracker.estimate(prompt_text, response_text)
        best_score = result.matches[0].score if result.matches else None

        history_service.log_prompt(
            HistoryEntry(
                task_type=TASK_RETRIEVAL,
                image_name=query_name,
                prompt=prompt_text,
                response=response_text,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                inference_time=result.inference_time,
            )
        )
        history_service.log_request(
            RequestLogEntry(
                request_type=REQUEST_TYPE_CLIP,
                task_type=TASK_RETRIEVAL,
                similarity_score=best_score,
                inference_time=result.inference_time,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        )
