"""CLIP model wrapper for image and text embeddings."""

from __future__ import annotations

import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import SETTINGS
from models.loader import resolve_device, torch_dtype_for_device
from utils.logger import get_logger

logger = get_logger(__name__)


class ClipEngine:
    """Wraps a Hugging Face CLIP model for embedding generation."""

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        self.model_id = model_id or SETTINGS.model.clip_model_id
        self.device = resolve_device(device)
        self.dtype = torch_dtype_for_device(self.device)

        logger.info("Loading CLIP model '%s' on device '%s'.", self.model_id, self.device)
        self.model = CLIPModel.from_pretrained(self.model_id, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        logger.info("CLIP model loaded successfully.")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Generate a normalized embedding vector for a single image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).to(torch.float32).cpu().numpy()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Generate normalized embedding vectors for a batch of images."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.to(torch.float32).cpu().numpy()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embedding vectors for a batch of text prompts."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.to(torch.float32).cpu().numpy()

    @torch.no_grad()
    def image_text_logits(self, image: Image.Image, texts: list[str]) -> np.ndarray:
        """Compute raw CLIP logits between one image and multiple text prompts."""
        inputs = self.processor(
            text=texts, images=image, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image.squeeze(0).to(torch.float32).cpu().numpy()
        return logits


@st.cache_resource(show_spinner="Loading CLIP model...")
def get_clip_engine(model_id: str | None = None, device: str | None = None) -> ClipEngine:
    """Return a cached CLIP engine instance for the given model and device."""
    return ClipEngine(model_id=model_id, device=device)
