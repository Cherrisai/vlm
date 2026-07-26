"""Application-wide constant values."""

from __future__ import annotations

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]

NAV_DASHBOARD = "Dashboard"
NAV_IMAGE_SIMILARITY = "Image Similarity"
NAV_IMAGE_TEXT_MATCHING = "Image Text Matching"
NAV_RETRIEVAL = "Image Retrieval"
NAV_CAPTIONING = "Image Captioning"
NAV_VISUAL_QA = "Visual QA"
NAV_IMAGE_ANALYSIS = "Image Analysis"
NAV_PROMPT_HISTORY = "Prompt History"
NAV_ANALYTICS = "Analytics"
NAV_SETTINGS = "Settings"
NAV_ABOUT = "About"

NAV_ITEMS = [
    NAV_DASHBOARD,
    NAV_IMAGE_SIMILARITY,
    NAV_IMAGE_TEXT_MATCHING,
    NAV_RETRIEVAL,
    NAV_CAPTIONING,
    NAV_VISUAL_QA,
    NAV_IMAGE_ANALYSIS,
    NAV_PROMPT_HISTORY,
    NAV_ANALYTICS,
    NAV_SETTINGS,
    NAV_ABOUT,
]

REQUEST_TYPE_CLIP = "clip"
REQUEST_TYPE_LLAVA = "llava"

TASK_SIMILARITY = "image_similarity"
TASK_TEXT_MATCHING = "image_text_matching"
TASK_RETRIEVAL = "image_retrieval"
TASK_CAPTIONING = "image_captioning"
TASK_VQA = "visual_qa"
TASK_ANALYSIS = "image_analysis"
TASK_OCR = "ocr_assistance"

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"

DEFAULT_ANALYSIS_PROMPT = (
    "Provide a structured analysis of this image. Describe the scene summary, "
    "the main objects present, the actions taking place, the environment, and "
    "a possible use case for this image."
)

OCR_PROMPT = (
    "Read any text visible in this image. Then explain what the text means, "
    "provide a concise summary of it, and extract the most important information."
)

CAPTION_PROMPT = (
    "Describe this image in detail, including the scene, the objects present, "
    "and the surrounding context."
)
