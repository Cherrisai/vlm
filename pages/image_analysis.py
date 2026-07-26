"""Structured image analysis dashboard and OCR assistance page."""

from __future__ import annotations

import streamlit as st

from components.cards import render_metric_row, render_result_card, render_section_header
from components.uploader import render_image_uploader
from models.llava_model import get_llava_engine
from services.analysis import AnalysisService
from services.history_service import HistoryEntry, HistoryService, RequestLogEntry
from services.token_tracker import TokenTracker
from utils.constants import (
    DEFAULT_ANALYSIS_PROMPT,
    OCR_PROMPT,
    REQUEST_TYPE_LLAVA,
    TASK_ANALYSIS,
    TASK_OCR,
)
from utils.session_settings import get_effective_device, get_effective_llava_model_id


def render(history_service: HistoryService, token_tracker: TokenTracker) -> None:
    """Render the structured image analysis dashboard with an OCR assistance tab."""
    render_section_header(
        "Image Analysis", "Generate a structured AI analysis report or extract text with OCR reasoning."
    )

    image, image_name = render_image_uploader("Upload an image", key="analysis_image")
    if image is not None:
        st.image(image, width=320)

    tab_analysis, tab_ocr = st.tabs(["Structured Analysis", "OCR Assistance"])

    with tab_analysis:
        if st.button("Run Analysis", type="primary", disabled=image is None, key="run_analysis"):
            with st.spinner("Analyzing image with LLaVA..."):
                llava_engine = get_llava_engine(
                    model_id=get_effective_llava_model_id(), device=get_effective_device()
                )
                analysis_service = AnalysisService(llava_engine)
                report = analysis_service.analyze_image(image)

            render_metric_row(
                [
                    ("Confidence", f"{report.confidence}%", None),
                    ("Inference Time", f"{report.inference_time} s", None),
                ]
            )
            render_result_card("Scene Summary", report.scene_summary)
            render_result_card("Objects", report.objects)
            render_result_card("Actions", report.actions)
            render_result_card("Environment", report.environment)
            render_result_card("Possible Use Case", report.use_case)
            render_result_card("Detailed AI Analysis", report.detailed_analysis)

            combined_response = (
                f"{report.scene_summary}\n{report.objects}\n{report.actions}\n"
                f"{report.environment}\n{report.use_case}\n{report.detailed_analysis}"
            )
            usage = token_tracker.estimate(DEFAULT_ANALYSIS_PROMPT, combined_response)

            history_service.log_prompt(
                HistoryEntry(
                    task_type=TASK_ANALYSIS,
                    image_name=image_name,
                    prompt=DEFAULT_ANALYSIS_PROMPT,
                    response=combined_response,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    inference_time=report.inference_time,
                )
            )
            history_service.log_request(
                RequestLogEntry(
                    request_type=REQUEST_TYPE_LLAVA,
                    task_type=TASK_ANALYSIS,
                    similarity_score=None,
                    inference_time=report.inference_time,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
            )

    with tab_ocr:
        if st.button("Run OCR Assistance", type="primary", disabled=image is None, key="run_ocr"):
            with st.spinner("Reading and reasoning over text with LLaVA..."):
                llava_engine = get_llava_engine(
                    model_id=get_effective_llava_model_id(), device=get_effective_device()
                )
                analysis_service = AnalysisService(llava_engine)
                ocr_result = analysis_service.run_ocr_assistance(image)

            render_result_card("Extracted Text and Explanation", ocr_result.extracted_explanation)
            render_result_card("Summary", ocr_result.summary)
            render_result_card("Translation", ocr_result.translation)
            render_result_card(
                "Important Information",
                ocr_result.important_information,
                footer=f"Inference time: {ocr_result.inference_time} s",
            )

            combined_response = (
                f"{ocr_result.extracted_explanation}\n{ocr_result.summary}\n"
                f"{ocr_result.translation}\n{ocr_result.important_information}"
            )
            usage = token_tracker.estimate(OCR_PROMPT, combined_response)

            history_service.log_prompt(
                HistoryEntry(
                    task_type=TASK_OCR,
                    image_name=image_name,
                    prompt=OCR_PROMPT,
                    response=combined_response,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    inference_time=ocr_result.inference_time,
                )
            )
            history_service.log_request(
                RequestLogEntry(
                    request_type=REQUEST_TYPE_LLAVA,
                    task_type=TASK_OCR,
                    similarity_score=None,
                    inference_time=ocr_result.inference_time,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
            )
