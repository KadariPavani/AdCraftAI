# AdCraft AI

> **⚠️ IMPORTANT:** This repository requires **Git LFS**. Install it before cloning:  
> `git lfs install` → [Installation Guide](https://git-lfs.github.com/)  
> **Having issues?** See [QUICK_FIX_FAISS.md](QUICK_FIX_FAISS.md)

AdCraft AI is a dataset-driven ad generation platform. It turns a product prompt and optional image into a polished ad creative, then lets you save, share, and track it through a public product hub.

## What it does

- Generate ad creatives from text prompts and optional images
- Parse free-form product descriptions into structured fields
- Generate product descriptions and multi-language captions
- Enhance uploaded images
- Save generated ads as products
- Share products through public hub pages
- Track engagement per platform
- Browse dataset stats, categories, and supported languages

## Main URLs

- App: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/health`
- Dataset explorer: `http://localhost:8000/dataset`

## Project layout

- `app/main.py` - FastAPI routes and HTML pages
- `app/pipeline.py` - Ad generation pipeline
- `app/smart_prompt.py` - Prompt parsing and field extraction
- `app/database.py` - SQLite product storage and analytics
- `app/image_gen.py` - Image generation with fallbacks
- `app/content_gen.py` - Text generation, translation, and enhancement
- `app/designer.py` - Layout composition
- `app/static/` - Frontend pages
- `data/annotations/` - Dataset metadata
- `embeddings/` - Prebuilt embeddings and search index
- `processed/` - Processed metadata
- `outputs/` - Generated creatives
- `uploads/` - User uploads
- `products_db/` - Local SQLite database
- `run.py` - Local server launcher
- `docker-compose.yml` - Docker Compose config
- `Dockerfile` - Container build

## Environment variables

Create a `.env` file in the project root.

Required for best results:

- `HF_TOKEN` - Hugging Face token for higher quality image generation
- `GOOGLE_API_KEY` - Gemini text generation

Optional:

- `GROQ_API_KEY`
- `TOGETHER_API_KEY`
- `ANTHROPIC_API_KEY`
- `HF_IMG2IMG_PROVIDERS` - Comma-separated provider order for FLUX img2img (default: `fal-ai,blackforestlabs,replicate,together,auto`)
- `PORT` - defaults to `8000` locally

## Local setup after clone

**⚠️ IMPORTANT: Git LFS is required for this repository!**

**→ Quick Start: [QUICKSTART.md](QUICKSTART.md) (5 steps)**  
**→ Full Guide: [COMPLETE_SETUP_COMMANDS.md](COMPLETE_SETUP_COMMANDS.md)**

```bash
# Quick version (copy-paste):
git lfs install
git clone <your-repo-url>
cd <repo-folder>
git lfs pull
python scripts/check_faiss.py  # Validate
copy .env.example .env
# Edit .env, add API keys, then:
docker-compose up -d --build
```

Open `http://localhost:8000`.

### Without Docker (Local Python):

```bash
# After git lfs pull and creating .env:
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python run.py
```

## Local setup notes

- The app loads environment variables from `.env`
- The server runs on port `8000` by default
- If `HF_TOKEN` is missing, image generation falls back to simpler output
- The app creates runtime folders automatically if they do not exist
- `outputs/`, `uploads/`, and `products_db/` are persistent local folders

## Docker setup after clone

```bash
git clone <your-repo-url>
cd <repo-folder>
copy .env.example .env
docker-compose up -d --build
```

Open:

- `http://localhost:8000`
- `http://localhost:8000/docs`

Stop Docker:

```bash
docker-compose down
```

View logs:

```bash
docker-compose logs -f
```

## Docker behavior

- The container reads `.env`
- `PORT=8000` is set in Compose
- `SKIP_DATASET_DOWNLOAD=1` is used so the app starts without downloading the original image archive
- `products_db/`, `outputs/`, and `uploads/` are mounted for persistence
- Health check hits `/api/health`
- Restart policy is `unless-stopped`

## API surface

### Pages

- `GET /` - main UI
- `GET /dataset` - dataset explorer
- `GET /hub/{product_id}` - public product hub

### Core API

- `GET /api/health`
- `GET /api/languages`
- `GET /api/stats`
- `GET /api/dataset-summary`
- `GET /api/category-fields`
- `GET /api/categories`
- `POST /api/parse-prompt`
- `POST /api/validate-fields`
- `POST /api/generate`
- `POST /api/save-generated`
- `POST /api/products`
- `GET /api/products`
- `GET /api/products/{product_id}`
- `DELETE /api/products/{product_id}`
- `POST /api/products/{product_id}/generate`
- `POST /api/enhance`
- `POST /api/describe`
- `POST /api/captions`
- `POST /api/track/{product_id}`
- `GET /api/analytics/{product_id}`

## Supported generation flow

1. Parse the prompt and detect product fields
2. Retrieve relevant dataset references
3. Build ad copy and captions
4. Compose the visual creative
5. Save the result as a product if needed
6. Share the product through the hub page

### `/api/generate` optional form fields

- `query` - Alias for `prompt` (either one is accepted)
- `model` - FLUX model preference:
  - `flux1-kontext-dev` (recommended for uploaded product image editing)
  - `flux1-redux-dev`
  - `flux2-dev`
  - `flux2-pro`
  - `flux2-max`
- `style` - Extra instruction to enforce ad style/quality in the pipeline

## Supported content

- Product title
- Product description
- Instagram caption
- WhatsApp copy
- Hashtags
- Multi-language translations
- Shareable public page
- Engagement tracking by platform

## Generated and runtime folders

These folders are part of normal runtime behavior and should stay in the repo:

- `embeddings/`
- `processed/`
- `data/annotations/`

These are generated at runtime:

- `outputs/`
- `uploads/`
- `products_db/`

## Common issues

| Problem | Fix |
|---|---|
| **FAISS index error** | Git LFS files not downloaded. Run: `git lfs install && git lfs pull` then rebuild Docker |
| App does not start | Check `.env` and install dependencies |
| Image quality is weak | Add `HF_TOKEN` |
| Text generation is limited | Add `GOOGLE_API_KEY` or another optional key |
| Port already in use | Stop the other process or change `PORT` |
| Docker restart loop | Check `docker-compose logs` |

**To validate your setup:** `python scripts/check_faiss.py`

## Quick commands

```bash
python run.py
docker-compose up -d --build
docker-compose logs -f
docker-compose down
```

