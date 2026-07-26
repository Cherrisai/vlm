"""Image loading, validation, and preprocessing helpers."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL image from raw bytes and normalize it to RGB."""
    image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")
    return image


def load_image_from_path(path: str | Path) -> Image.Image:
    """Load a PIL image from a filesystem path and normalize it to RGB."""
    image = Image.open(path)
    image = image.convert("RGB")
    return image


def resize_image(image: Image.Image, target_size: int) -> Image.Image:
    """Resize an image to a square target size while preserving aspect ratio via padding."""
    width, height = image.size
    scale = target_size / max(width, height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = image.resize((new_width, new_height), Image.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def compute_image_hash(data: bytes) -> str:
    """Compute a stable SHA-256 hash for image bytes, used for caching and dedup."""
    return hashlib.sha256(data).hexdigest()


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize a PIL image back to raw bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def make_thumbnail(image: Image.Image, size: tuple[int, int] = (160, 160)) -> Image.Image:
    """Create a thumbnail copy of an image without mutating the original."""
    thumb = image.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    return thumb


def list_dataset_images(directory: Path) -> list[Path]:
    """List all supported image files within a directory."""
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions
    )
