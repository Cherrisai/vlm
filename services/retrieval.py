"""Image retrieval service: finds top-K similar images from a dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import SETTINGS
from models.clip_model import ClipEngine
from utils.image_utils import list_dataset_images, load_image_from_path
from utils.metrics import cosine_similarity, timer
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalMatch:
    rank: int
    image_path: Path
    score: float


@dataclass
class RetrievalResult:
    matches: list[RetrievalMatch]
    dataset_size: int
    inference_time: float


class RetrievalService:
    """Performs CLIP-embedding based nearest neighbor image retrieval."""

    def __init__(self, clip_engine: ClipEngine) -> None:
        self.clip_engine = clip_engine

    def retrieve(
        self, query_image: Image.Image, dataset_dir: Path | None = None, top_k: int = 5
    ) -> RetrievalResult:
        """Retrieve the top-K most similar images to the query from a dataset directory."""
        directory = dataset_dir or SETTINGS.retrieval_dataset_dir
        candidate_paths = list_dataset_images(directory)

        if not candidate_paths:
            return RetrievalResult(matches=[], dataset_size=0, inference_time=0.0)

        with timer() as t:
            query_embedding = self.clip_engine.encode_image(query_image)

            scored: list[RetrievalMatch] = []
            for path in candidate_paths:
                try:
                    candidate_image = load_image_from_path(path)
                    candidate_embedding = self.clip_engine.encode_image(candidate_image)
                    score = cosine_similarity(query_embedding, candidate_embedding)
                    scored.append(RetrievalMatch(rank=0, image_path=path, score=round(score, 4)))
                except Exception:
                    logger.exception("Failed to process dataset image %s", path)

            scored.sort(key=lambda m: m.score, reverse=True)
            top_matches = scored[:top_k]
            for idx, match in enumerate(top_matches, start=1):
                match.rank = idx

        return RetrievalResult(
            matches=top_matches, dataset_size=len(candidate_paths), inference_time=t.elapsed_seconds
        )
