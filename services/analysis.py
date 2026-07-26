"""Image analysis service: structured reports, visual QA, and OCR reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from models.llava_model import LlavaEngine
from utils.constants import DEFAULT_ANALYSIS_PROMPT, OCR_PROMPT
from utils.metrics import timer
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisReport:
    scene_summary: str
    objects: str
    actions: str
    environment: str
    use_case: str
    detailed_analysis: str
    confidence: float
    inference_time: float


@dataclass
class VQATurn:
    question: str
    answer: str


@dataclass
class OCRResult:
    extracted_explanation: str
    summary: str
    translation: str
    important_information: str
    inference_time: float


class AnalysisService:
    """Provides structured image analysis, visual question answering, and OCR."""

    def __init__(self, llava_engine: LlavaEngine) -> None:
        self.llava_engine = llava_engine

    def analyze_image(self, image: Image.Image) -> AnalysisReport:
        """Produce a structured multi-field analysis report for an image."""
        with timer() as t:
            scene_summary = self.llava_engine.generate(
                image, "Summarize the scene in this image in one to two sentences."
            )
            objects = self.llava_engine.generate(
                image, "List the key objects visible in this image."
            )
            actions = self.llava_engine.generate(
                image, "Describe any actions or activities taking place in this image."
            )
            environment = self.llava_engine.generate(
                image, "Describe the environment or setting shown in this image."
            )
            use_case = self.llava_engine.generate(
                image, "Suggest a plausible use case or purpose for this image."
            )
            detailed_analysis = self.llava_engine.generate(image, DEFAULT_ANALYSIS_PROMPT)

        word_count = len(detailed_analysis.split())
        confidence = round(min(99.0, 55.0 + min(word_count, 90) * 0.4), 2)

        return AnalysisReport(
            scene_summary=scene_summary,
            objects=objects,
            actions=actions,
            environment=environment,
            use_case=use_case,
            detailed_analysis=detailed_analysis,
            confidence=confidence,
            inference_time=t.elapsed_seconds,
        )

    def ask_question(
        self, image: Image.Image, question: str, history: list[VQATurn] | None = None
    ) -> tuple[str, float]:
        """Answer a visual question, optionally conditioned on prior conversation turns."""
        history_payload = [{"question": h.question, "answer": h.answer} for h in (history or [])]
        with timer() as t:
            answer = self.llava_engine.chat(image, history_payload, question)
        return answer, t.elapsed_seconds

    def run_ocr_assistance(self, image: Image.Image) -> OCRResult:
        """Extract, explain, summarize, and translate any text present in an image."""
        with timer() as t:
            explanation = self.llava_engine.generate(image, OCR_PROMPT)
            summary = self.llava_engine.generate(
                image, "Summarize any text found in this image in one short paragraph."
            )
            translation = self.llava_engine.generate(
                image,
                "If there is text in this image that is not in English, translate it to English. "
                "If it is already in English or there is no text, state that clearly.",
            )
            important_info = self.llava_engine.generate(
                image, "Extract the most important pieces of information from any text in this image."
            )

        return OCRResult(
            extracted_explanation=explanation,
            summary=summary,
            translation=translation,
            important_information=important_info,
            inference_time=t.elapsed_seconds,
        )
