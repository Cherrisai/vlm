"""Image-Image similarity page powered by CLIP."""

from __future__ import annotations

import streamlit as st

from components.cards import render_metric_row, render_result_card, render_section_header
from components.uploader import render_image_uploader
from models.clip_model import get_clip_engine
from services.history_service import HistoryEntry, HistoryService, RequestLogEntry
from services.similarity import SimilarityService
from services.token_tracker import TokenTracker
from utils.constants import REQUEST_TYPE_CLIP, TASK_SIMILARITY
from utils.session_settings import get_effective_clip_model_id, get_effective_device


def render(history_service: HistoryService, token_tracker: TokenTracker) -> None:
    """Render the image-image similarity comparison page."""
    render_section_header(
        "Image-Image Similarity", "Compare two images using CLIP embeddings and cosine similarity."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        image_a, name_a = render_image_uploader("Upload first image", key="similarity_image_a")
        if image_a is not None:
            st.image(image_a, width=320)
    with col_b:
        image_b, name_b = render_image_uploader("Upload second image", key="similarity_image_b")
        if image_b is not None:
            st.image(image_b, width=320)

    if st.button("Compute Similarity", type="primary", disabled=image_a is None or image_b is None):
        clip_engine = get_clip_engine(
            model_id=get_effective_clip_model_id(), device=get_effective_device()
        )
        similarity_service = SimilarityService(clip_engine)
        result = similarity_service.compare_images(image_a, image_b)

        render_metric_row(
            [
                ("Cosine Similarity", f"{result.cosine_similarity}", None),
                ("Similarity Percentage", f"{result.similarity_percentage}%", None),
                ("Confidence", f"{result.confidence}%", None),
                ("Embedding Distance", f"{result.embedding_distance}", None),
            ]
        )

        render_result_card(
            title="Result Summary",
            content=(
                f"The two images share a cosine similarity of {result.cosine_similarity}, "
                f"corresponding to a confidence level of {result.confidence}%. "
                f"The Euclidean embedding distance is {result.embedding_distance}."
            ),
            footer=f"Inference time: {result.inference_time} s",
        )

        st.progress(min(1.0, max(0.0, result.confidence / 100)))

        prompt_text = f"{name_a} vs {name_b}"
        response_text = f"cosine_similarity={result.cosine_similarity}"
        usage = token_tracker.estimate(prompt_text, response_text)

        history_service.log_prompt(
            HistoryEntry(
                task_type=TASK_SIMILARITY,
                image_name=f"{name_a} | {name_b}",
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
                task_type=TASK_SIMILARITY,
                similarity_score=result.cosine_similarity,
                inference_time=result.inference_time,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        )
