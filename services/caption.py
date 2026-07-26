"""Captioning service built on the LLaVA engine."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from models.llava_model import LlavaEngine
from utils.constants import CAPTION_PROMPT
from utils.metrics import timer
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CaptionResult:
    detailed_caption: str
    scene_description: str
    object_description: str
    context: str
    inference_time: float


class CaptionService:
    """Generates structured captions for an image using LLaVA reasoning."""

    def __init__(self, llava_engine: LlavaEngine) -> None:
        self.llava_engine = llava_engine

    def generate_caption(self, image: Image.Image) -> CaptionResult:
        """Generate a detailed caption plus scene, object, and context breakdowns."""
        with timer() as t:
            detailed_caption = self.llava_engine.generate(image, CAPTION_PROMPT)
            scene_description = self.llava_engine.generate(
                image, "Describe only the overall scene and setting of this image in two sentences."
            )
            object_description = self.llava_engine.generate(
                image, "List and briefly describe the main objects visible in this image."
            )
            context = self.llava_engine.generate(
                image, "What is the likely context or situation depicted in this image?"
            )

        return CaptionResult(
            detailed_caption=detailed_caption,
            scene_description=scene_description,
            object_description=object_description,
            context=context,
            inference_time=t.elapsed_seconds,
        )
