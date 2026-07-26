"""Image-Text matching page powered by CLIP."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import render_result_card, render_section_header
from components.charts import render_bar_chart, render_probability_table
from utils.metrics import similarity_to_confidence
from components.uploader import render_image_uploader
from models.clip_model import get_clip_engine
from services.history_service import HistoryEntry, HistoryService, RequestLogEntry
from services.similarity import SimilarityService
from services.token_tracker import TokenTracker
from utils.constants import REQUEST_TYPE_CLIP, TASK_TEXT_MATCHING
from utils.session_settings import get_effective_clip_model_id, get_effective_device


def render(history_service: HistoryService, token_tracker: TokenTracker) -> None:
    """Render the image-text matching page."""
    render_section_header(
        "Image-Text Matching", "Rank multiple text prompts against an uploaded image using CLIP."
    )

    image, image_name = render_image_uploader("Upload an image", key="itm_image")
    if image is not None:
        st.image(image, width=320)

    prompts_raw = st.text_area(
        "Enter text prompts, one per line",
        placeholder="a photo of a dog\na photo of a cat\na person riding a bicycle",
        height=140,
    )
    prompts = [p.strip() for p in prompts_raw.splitlines() if p.strip()]

    if st.button("Match Prompts", type="primary", disabled=image is None or not prompts):
        clip_engine = get_clip_engine(
            model_id=get_effective_clip_model_id(), device=get_effective_device()
        )
        similarity_service = SimilarityService(clip_engine)
        result = similarity_service.match_image_to_texts(image, prompts)

        top_confidence = similarity_to_confidence(result.top_match.similarity)
        render_result_card(
            title="Top Match",
            content=(
                f"\"{result.top_match.prompt}\" ranked highest among the prompts provided "
                f"({result.top_match.probability}% relative probability), with a cosine "
                f"similarity of {result.top_match.similarity} (absolute confidence: "
                f"{top_confidence}%)."
            ),
            footer=(
                f"Inference time: {result.inference_time} s. "
                f"Note: relative probability is only meaningful with 2+ prompts."
            ),
        )

      
        labels = [m.prompt for m in result.matches]
        probabilities = [m.probability for m in result.matches]
        render_bar_chart(labels, probabilities, "Prompt Probability Ranking", "Probability (%)")

        df = pd.DataFrame(
            [
                {"Prompt": m.prompt, "Similarity": m.similarity, "Probability (%)": m.probability}
                for m in result.matches
            ]
        )
        render_probability_table(df)

        prompt_text = " | ".join(prompts)
        response_text = f"top_match={result.top_match.prompt}"
        usage = token_tracker.estimate(prompt_text, response_text)

        history_service.log_prompt(
            HistoryEntry(
                task_type=TASK_TEXT_MATCHING,
                image_name=image_name,
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
                task_type=TASK_TEXT_MATCHING,
                similarity_score=result.top_match.similarity,
                inference_time=result.inference_time,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        )
