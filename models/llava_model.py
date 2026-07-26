"""Vision-language model wrapper for captioning, VQA, and analysis.

Supports two interchangeable backends behind the same generate()/chat() interface:

- LlavaEngine: full LLaVA models (llava-hf/llava-1.5-7b-hf and similar). Strong quality,
  but the weights are ~13-15GB and generation is slow without a GPU.
- MoondreamEngine: vikhyatk/moondream2, a much smaller (~3.9GB) vision-language model
  that runs acceptably on CPU and is a practical choice for local development, low-bandwidth
  connections, or free-tier Hugging Face Spaces deployments.

The active backend is selected automatically based on the configured model id, so the rest
of the application (services, pages) never needs to know which backend is loaded.
"""

from __future__ import annotations

import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

from config import SETTINGS
from models.loader import resolve_device, torch_dtype_for_device
from utils.logger import get_logger

logger = get_logger(__name__)

MOONDREAM_REVISION = "2024-08-26"


class LlavaEngine:
    """Wraps a Hugging Face LLaVA model for multimodal text generation."""

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        self.model_id = model_id or SETTINGS.model.llava_model_id
        self.device = resolve_device(device)
        self.dtype = torch_dtype_for_device(self.device)

        logger.info("Loading LLaVA model '%s' on device '%s'.", self.model_id, self.device)
        load_kwargs = {"torch_dtype": self.dtype}
        if SETTINGS.model.load_in_4bit and self.device == "cuda":
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
        if not SETTINGS.model.load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info("LLaVA model loaded successfully.")

    def _build_prompt(self, instruction: str) -> str:
        """Build a LLaVA-compatible chat prompt embedding the image placeholder."""
        return f"USER: <image>\n{instruction}\nASSISTANT:"

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        instruction: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """Generate a text response conditioned on an image and an instruction."""
        prompt = self._build_prompt(instruction)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, self.dtype if self.device == "cuda" else torch.float32
        )

        generation_kwargs = dict(
            max_new_tokens=max_new_tokens or SETTINGS.model.max_new_tokens,
            do_sample=True,
            temperature=temperature or SETTINGS.model.temperature,
            top_p=top_p or SETTINGS.model.top_p,
        )

        output_ids = self.model.generate(**inputs, **generation_kwargs)
        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        if "ASSISTANT:" in decoded:
            decoded = decoded.split("ASSISTANT:", 1)[1].strip()
        return decoded.strip()

    @torch.no_grad()
    def chat(
        self,
        image: Image.Image,
        conversation_history: list[dict],
        question: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate a response for a follow-up question, incorporating prior turns."""
        history_text = ""
        for turn in conversation_history[-5:]:
            history_text += f"USER: {turn['question']}\nASSISTANT: {turn['answer']}\n"
        instruction = f"{history_text}{question}" if history_text else question
        return self.generate(image, instruction, max_new_tokens=max_new_tokens)


class MoondreamEngine:
    """Wraps the lightweight vikhyatk/moondream2 vision-language model.

    Moondream2 exposes its own encode_image/answer_question API (via trust_remote_code)
    rather than the standard generate() pipeline used by LLaVA, so this class adapts it
    to the same generate()/chat() interface used elsewhere in the application.

    The image encoding step is cached per image object so that pages issuing several
    prompts against the same image (captioning, structured analysis) only pay the
    vision-encoding cost once instead of once per prompt.
    """

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        self.model_id = model_id or "vikhyatk/moondream2"
        self.device = resolve_device(device)
        self.dtype = torch_dtype_for_device(self.device)

        logger.info("Loading Moondream2 model '%s' on device '%s'.", self.model_id, self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=MOONDREAM_REVISION,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=MOONDREAM_REVISION)
        logger.info("Moondream2 model loaded successfully.")

        self._cached_image_id: int | None = None
        self._cached_encoding = None

    def _get_encoded_image(self, image: Image.Image):
        if self._cached_image_id != id(image):
            self._cached_encoding = self.model.encode_image(image)
            self._cached_image_id = id(image)
        return self._cached_encoding

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        instruction: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """Generate a text response conditioned on an image and an instruction."""
        encoded_image = self._get_encoded_image(image)
        answer = self.model.answer_question(encoded_image, instruction, self.tokenizer)
        return answer.strip()

    @torch.no_grad()
    def chat(
        self,
        image: Image.Image,
        conversation_history: list[dict],
        question: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate a response for a follow-up question, incorporating prior turns."""
        history_text = ""
        for turn in conversation_history[-5:]:
            history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n"
        instruction = f"{history_text}Q: {question}" if history_text else question
        return self.generate(image, instruction, max_new_tokens=max_new_tokens)


def _is_moondream(model_id: str) -> bool:
    return "moondream" in model_id.lower()


@st.cache_resource(show_spinner="Loading vision-language model...")
def get_llava_engine(model_id: str | None = None, device: str | None = None):
    """Return a cached VLM engine instance, dispatching to LLaVA or Moondream2 by model id."""
    resolved_model_id = model_id or SETTINGS.model.llava_model_id
    if _is_moondream(resolved_model_id):
        return MoondreamEngine(model_id=resolved_model_id, device=device)
    return LlavaEngine(model_id=resolved_model_id, device=device)
