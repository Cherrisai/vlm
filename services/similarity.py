"""Similarity service for image-image and image-text matching using CLIP."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from models.clip_model import ClipEngine
from utils.metrics import cosine_similarity, euclidean_distance, similarity_to_confidence, softmax, timer
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImageSimilarityResult:
    cosine_similarity: float
    similarity_percentage: float
    confidence: float
    embedding_distance: float
    inference_time: float


@dataclass
class TextMatchResult:
    prompt: str
    similarity: float
    probability: float


@dataclass
class ImageTextMatchingResult:
    matches: list[TextMatchResult]
    top_match: TextMatchResult
    inference_time: float


class SimilarityService:
    """Provides CLIP-based similarity computations."""

    def __init__(self, clip_engine: ClipEngine) -> None:
        self.clip_engine = clip_engine

    def compare_images(self, image_a: Image.Image, image_b: Image.Image) -> ImageSimilarityResult:
        """Compare two images and return similarity, confidence, and distance metrics."""
        with timer() as t:
            embedding_a = self.clip_engine.encode_image(image_a)
            embedding_b = self.clip_engine.encode_image(image_b)
            similarity = cosine_similarity(embedding_a, embedding_b)
            distance = euclidean_distance(embedding_a, embedding_b)

        return ImageSimilarityResult(
            cosine_similarity=round(similarity, 4),
            similarity_percentage=round(max(0.0, similarity) * 100, 2),
            confidence=similarity_to_confidence(similarity),
            embedding_distance=round(distance, 4),
            inference_time=t.elapsed_seconds,
        )

    def match_image_to_texts(
        self, image: Image.Image, prompts: list[str]
    ) -> ImageTextMatchingResult:
        """Rank a set of text prompts against an image using CLIP similarity."""
        if not prompts:
            raise ValueError("At least one text prompt is required.")

        with timer() as t:
            logits = self.clip_engine.image_text_logits(image, prompts)
            probabilities = softmax(logits)

            image_embedding = self.clip_engine.encode_image(image)
            text_embeddings = self.clip_engine.encode_text(prompts)

            results = []
            for idx, prompt in enumerate(prompts):
                similarity = cosine_similarity(image_embedding, text_embeddings[idx])
                results.append(
                    TextMatchResult(
                        prompt=prompt,
                        similarity=round(similarity, 4),
                        probability=round(float(probabilities[idx]) * 100, 2),
                    )
                )

        results.sort(key=lambda r: r.probability, reverse=True)
        return ImageTextMatchingResult(
            matches=results, top_match=results[0], inference_time=t.elapsed_seconds
        )
