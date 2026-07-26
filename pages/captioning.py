"""Image captioning page powered by LLaVA."""

from __future__ import annotations

import streamlit as st

from components.cards import render_result_card, render_section_header
from components.uploader import render_image_uploader
from models.llava_model import get_llava_engine
from services.caption import CaptionService
from services.history_service import HistoryEntry, HistoryService, RequestLogEntry
from services.token_tracker import TokenTracker
from utils.constants import CAPTION_PROMPT, REQUEST_TYPE_LLAVA, TASK_CAPTIONING
from utils.session_settings import get_effective_device, get_effective_llava_model_id


def render(history_service: HistoryService, token_tracker: TokenTracker) -> None:
    """Render the image captioning page."""
    render_section_header(
        "Image Captioning", "Generate detailed captions, scene, object, and context descriptions."
    )

    image, image_name = render_image_uploader("Upload an image", key="caption_image")
    if image is not None:
        st.image(image, width=320)

    if st.button("Generate Caption", type="primary", disabled=image is None):
        with st.spinner("Generating caption with LLaVA..."):
            llava_engine = get_llava_engine(
                model_id=get_effective_llava_model_id(), device=get_effective_device()
            )
            caption_service = CaptionService(llava_engine)
            result = caption_service.generate_caption(image)

        render_result_card("Detailed Caption", result.detailed_caption)
        render_result_card("Scene Description", result.scene_description)
        render_result_card("Object Description", result.object_description)
        render_result_card(
            "Context", result.context, footer=f"Inference time: {result.inference_time} s"
        )

        combined_response = (
            f"{result.detailed_caption}\n{result.scene_description}\n"
            f"{result.object_description}\n{result.context}"
        )
        usage = token_tracker.estimate(CAPTION_PROMPT, combined_response)

        history_service.log_prompt(
            HistoryEntry(
                task_type=TASK_CAPTIONING,
                image_name=image_name,
                prompt=CAPTION_PROMPT,
                response=combined_response,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                inference_time=result.inference_time,
            )
        )
        history_service.log_request(
            RequestLogEntry(
                request_type=REQUEST_TYPE_LLAVA,
                task_type=TASK_CAPTIONING,
                similarity_score=None,
                inference_time=result.inference_time,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        )
