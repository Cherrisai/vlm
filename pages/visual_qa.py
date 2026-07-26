"""Visual Question Answering page powered by LLaVA."""

from __future__ import annotations

import streamlit as st

from components.cards import render_section_header
from components.uploader import render_image_uploader
from models.llava_model import get_llava_engine
from services.analysis import AnalysisService, VQATurn
from services.history_service import HistoryEntry, HistoryService, RequestLogEntry
from services.token_tracker import TokenTracker
from utils.constants import REQUEST_TYPE_LLAVA, TASK_VQA
from utils.session_settings import get_effective_device, get_effective_llava_model_id

SESSION_KEY = "vqa_conversation"


def render(history_service: HistoryService, token_tracker: TokenTracker) -> None:
    """Render the visual question answering page with persistent conversation history."""
    render_section_header(
        "Visual Question Answering", "Ask questions about an uploaded image using LLaVA."
    )

    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = []

    image, image_name = render_image_uploader("Upload an image", key="vqa_image")
    if image is not None:
        st.image(image, width=320)

    for turn in st.session_state[SESSION_KEY]:
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            st.write(turn.answer)

    question = st.chat_input("Ask a question about the image, e.g. What is happening?")

    if question:
        if image is None:
            st.warning("Upload an image before asking a question.")
        else:
            llava_engine = get_llava_engine(
                model_id=get_effective_llava_model_id(), device=get_effective_device()
            )
            analysis_service = AnalysisService(llava_engine)
            history = st.session_state[SESSION_KEY]

            with st.spinner("Thinking..."):
                answer, inference_time = analysis_service.ask_question(image, question, history)

            history.append(VQATurn(question=question, answer=answer))

            usage = token_tracker.estimate(question, answer)
            history_service.log_prompt(
                HistoryEntry(
                    task_type=TASK_VQA,
                    image_name=image_name,
                    prompt=question,
                    response=answer,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    inference_time=inference_time,
                )
            )
            history_service.log_request(
                RequestLogEntry(
                    request_type=REQUEST_TYPE_LLAVA,
                    task_type=TASK_VQA,
                    similarity_score=None,
                    inference_time=inference_time,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
            )
            st.rerun()

    if st.session_state[SESSION_KEY] and st.button("Clear Conversation"):
        st.session_state[SESSION_KEY] = []
        st.rerun()
