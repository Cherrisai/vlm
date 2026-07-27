# Vision Intelligence Studio

A production-quality Vision Language Model (VLM) workspace built with Streamlit, combining
**OpenAI CLIP** (embedding-based similarity, matching, and retrieval) and **LLaVA** (open-ended
visual reasoning, captioning, question answering, and OCR) into a single, modular application.

Copyright (c) Saivignesh

---

## Project Overview

Vision Intelligence Studio is designed as an interview-ready and portfolio-ready reference
implementation of a modern computer vision / VLM product. It demonstrates clean architecture,
production-grade Python, and a professional Streamlit dashboard covering the full lifecycle of a
vision-language application: embeddings, generation, persistence, analytics, and observability.

## Features

**CLIP-powered**
- Image-Image Similarity: cosine similarity, similarity percentage, confidence, embedding distance
- Image-Text Matching: multi-prompt ranking with probabilities, bar chart, and probability table
- Image Retrieval: Top-K nearest neighbor search against a local image dataset

**LLaVA-powered**
- Image Captioning: detailed caption, scene description, object description, context
- Visual Question Answering: multi-turn conversation with history
- Image Analysis Dashboard: structured report (scene, objects, actions, environment, use case, confidence)
- OCR Assistance: explain, summarize, translate, and extract key information from text in images

**Platform**
- Prompt History: SQLite-backed, searchable, filterable, exportable to CSV, deletable
- Session Analytics: request counts, response times, similarity distribution, request timeline
- Token Dashboard: estimated input/output/total/conversation tokens with history and charts
- Performance Monitor: inference time, CPU/memory usage, GPU status, per-task timing
- Settings: model selection, device (CPU/CUDA/auto), image size, temperature, top-p, max tokens,
  batch size, theme, and cache clearing

## Architecture

```
project/
├── app.py                     Main Streamlit entrypoint and page router
├── config.py                  Centralized environment-driven configuration
├── requirements.txt
├── README.md
├── assets/
├── models/
│   ├── clip_model.py          CLIP embedding engine (cached resource)
│   ├── llava_model.py         LLaVA generation engine (cached resource)
│   └── loader.py               Device detection and dtype resolution
├── services/
│   ├── similarity.py          Image-image and image-text similarity logic
│   ├── retrieval.py           Top-K dataset retrieval logic
│   ├── caption.py             Captioning orchestration
│   ├── analysis.py            Structured analysis, VQA, and OCR orchestration
│   ├── token_tracker.py       Token estimation
│   └── history_service.py     Persistence and analytics queries
├── database/
│   ├── sqlite.py               Connection management and schema bootstrap
│   └── schema.py               SQL DDL statements
├── utils/
│   ├── image_utils.py
│   ├── metrics.py
│   ├── logger.py
│   ├── constants.py
│   ├── helpers.py
│   └── session_settings.py     Session-scoped settings resolution
├── components/
│   ├── sidebar.py
│   ├── cards.py
│   ├── charts.py
│   ├── uploader.py
│   ├── history_table.py
│   └── navbar.py
├── pages/
│   ├── dashboard.py
│   ├── image_similarity.py
│   ├── image_text_matching.py
│   ├── retrieval.py
│   ├── captioning.py
│   ├── visual_qa.py
│   ├── image_analysis.py
│   ├── analytics.py
│   ├── history.py
│   └── settings.py
└── data/                       SQLite database and local retrieval dataset
```

The codebase follows clean architecture principles: `pages/` handle presentation only, `services/`
contain business logic, `models/` wrap ML engines, `database/` isolates persistence, and `utils/`
provides shared, side-effect-free helpers.

## Installation

Requires Python 3.10 or later.

```bash
git clone <your-repository-url>
cd vision-intelligence-studio
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

On first use of Image Captioning, Visual QA, Image Analysis, or OCR Assistance, the LLaVA model
weights (several GB) will be downloaded from the Hugging Face Hub and cached locally. A GPU is
strongly recommended for LLaVA; CLIP runs comfortably on CPU.

### Configuration

All configuration is environment-driven (see `config.py`). Common overrides:

```bash
export VIS_CLIP_MODEL_ID="openai/clip-vit-base-patch32"
export VIS_LLAVA_MODEL_ID="llava-hf/llava-1.5-7b-hf"
export VIS_DEVICE="auto"        # auto | cpu | cuda
export VIS_LOAD_IN_4BIT="true"  # reduces GPU memory footprint for LLaVA
```

These can also be changed at runtime from the in-app Settings page.

## Deploy to Hugging Face Spaces

1. Create a new Space with the **Streamlit** SDK.
2. Push this repository's contents to the Space repository.
3. Ensure `requirements.txt` is present at the repository root (it is).
4. Set the Space hardware tier to a GPU tier if you intend to use LLaVA features; the CLIP-only
   features run on the free CPU tier.
5. The Space will automatically run `streamlit run app.py`.

No additional configuration files are required beyond what is included in this repository.

## Screenshots

_Add screenshots of the Dashboard, Image Similarity, Visual QA, and Analytics pages here before
publishing to GitHub or Hugging Face Spaces._

## Future Improvements

- Add authentication and per-user history isolation
- Support additional open-source VLMs (Qwen-VL, InternVL, Idefics)
- Add vector database backend (FAISS or Chroma) for large-scale retrieval
- Add streaming token-by-token generation for LLaVA responses
- Add automated evaluation harness for caption and VQA quality

## License



## Author

**Saivignesh**
Copyright (c) Saivignesh
# vlm
