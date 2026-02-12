# MAdVerse: RAG-Based Ad Generation Pipeline

A Retrieval-Augmented Generation pipeline that uses 61,576 real advertisement images (492 brands, 12 categories) to generate professional ad creatives. Real ads serve as context for the LLM, so generated output inherits the visual style, messaging, and branding conventions of actual professional advertisements.

---

## Project Structure

```
MAdVerse/
├── app/
│   ├── main.py              # FastAPI backend (API endpoints)
│   ├── pipeline.py           # Core ad generation pipeline (v2)
│   └── static/
│       └── index.html        # Web frontend UI
├── data/
│   ├── annotations/          # JSON metadata (13 MB)
│   └── images/               # 61,576 ad images (~35 GB)
│       ├── Advert_Gallery/
│       ├── OnlineAds/
│       ├── Epaper1/
│       └── Epaper2/
├── embeddings/
│   ├── image_embeddings.pkl  # CLIP embeddings (281 MB)
│   └── faiss_indexes/        # FAISS index + metadata (130 MB)
├── processed/
│   └── metadata/
│       └── madverse_metadata.csv  # Dataset metadata (61,592 rows)
├── outputs/                  # Generated ads (auto-created)
├── uploads/                  # User uploads (auto-created)
├── products_db/              # SQLite product catalog (auto-created)
├── DatasetLoad.py            # Step 1: Scan images → metadata CSV
├── GenerateEmbeddings.py     # Step 2: Images → CLIP embeddings
├── BuildFAISS.py             # Step 3: Embeddings → FAISS index
├── RAGPipeline.py            # Legacy pipeline (paid APIs)
├── run.py                    # Launch the web app
├── requirements.txt          # Python dependencies
└── .env                      # API keys (optional, not needed for v2)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web app
python run.py
```

Open **http://localhost:8000** in your browser.

> No API keys required — the current pipeline uses free services (Pollinations.ai).

---

## One-Time Setup (Index Building)

Only needed if rebuilding from scratch. Pre-built indexes are included.

**Step 1** — `python DatasetLoad.py`
- Scans 4 source folders (Advert_Gallery, OnlineAds, Epaper1, Epaper2)
- Extracts category/subcategory/brand from JSON annotations
- Outputs: `processed/metadata/madverse_metadata.csv` (61,592 rows, 492 brands)

**Step 2** — `python GenerateEmbeddings.py`
- Loads all 61,576 images through CLIP (`openai/clip-vit-base-patch32`)
- Generates L2-normalized 512-D embeddings per image
- Outputs: `embeddings/image_embeddings.pkl` (281 MB)
- Includes checkpoint system for resuming interrupted runs

**Step 3** — `python BuildFAISS.py`
- Loads embeddings, builds FAISS `IndexFlatL2(512)` index
- Outputs: `embeddings/faiss_indexes/madverse_index.faiss` + `id_to_metadata.pkl`

---

## Web App Features

The FastAPI web application (`run.py`) provides:

- **Ad Generation** — Text prompt + optional image → full ad creative
- **Product Catalog** — CRUD operations for managing products
- **Image Enhancement** — Brightness, contrast, sharpness adjustments
- **Multi-Language Captions** — 25+ languages via deep_translator
- **Shareable Hub Pages** — Public product pages with analytics tracking

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/generate` | POST | Generate ad from text + optional image |
| `/api/products` | GET/POST | Product catalog CRUD |
| `/api/enhance` | POST | Image enhancement |
| `/api/captions` | POST | Multi-language caption generation |
| `/hub/{product_id}` | GET | Public shareable product page |

---

## Pipeline Stages

### Stage 0 — Brand Matching
- Fuzzy-matches query text against 492 known brands
- Identifies target brand for filtered retrieval

### Stage 1 — RAG Retrieval (CLIP + FAISS)
- Encodes query text into a 512-D CLIP embedding
- Builds a brand-filtered sub-index from FAISS
- Retrieves top 5 most similar ad images from the dataset
- Extracts dominant colors via KMeans clustering

### Stage 2 — Content Generation (Pollinations.ai LLM)
- Generates taglines, feature highlights, and ad copy
- Crafts a detailed diffusion prompt for image generation

### Stage 3 — Image Generation (Pollinations.ai Diffusion)
- Generates a new product image from the LLM-crafted prompt
- Free, no API key required

### Stage 4 — Pamphlet Compositing
- Full-bleed gradient background using brand colors
- Hero product image with rounded corners + shadow
- Bold brand name, tagline, feature highlights, CTA button
- Reference thumbnails from retrieved dataset ads
- Output: 1080x1080 PNG saved to `outputs/`

---

## Architecture Flow

```
Query Text ──→ CLIP Embedding ──→ FAISS Search ──→ Top 5 Real Ads
                                                        │
                                                        ▼
                                                  Color Extraction
                                                  (KMeans clustering)
                                                        │
                                                        ▼
                                                Pollinations.ai LLM
                                                generates ad copy +
                                                  diffusion prompt
                                                        │
                                                        ▼
                                              Pollinations.ai Diffusion
                                                generates product image
                                                        │
                                                        ▼
                                                PamphletComposer
                                                creates final ad
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for CLIP embedding generation only)
- MAdVerse dataset (61,576 ad images with JSON annotations)
- No API keys needed — uses free Pollinations.ai services

### Key Dependencies

```
torch, torchvision       # PyTorch (CLIP model)
transformers              # HuggingFace (CLIP)
faiss-cpu                 # Vector search
fastapi, uvicorn          # Web server
Pillow, opencv-python     # Image processing
deep-translator           # Multi-language support
scikit-learn              # KMeans clustering
```
