# MAdVerse: RAG-Based Ad Generation Pipeline

A Retrieval-Augmented Generation pipeline that uses 61,576 real advertisement images (492 brands, 12 categories) to generate professional ad creatives. Real ads serve as context for the LLM, so generated output inherits the visual style, messaging, and branding conventions of actual professional advertisements.

---

## Essential Files

| File | Purpose |
|------|---------|
| `.env` | API keys (Google, HuggingFace, Anthropic) |
| `01_load_dataset.py` | Scan images → metadata CSV |
| `02_generate_embeddings.py` | Images → CLIP 512-D embeddings |
| `03_build_faiss_index.py` | Embeddings → FAISS vector index |
| `09_pamphlet_pipeline.py` | End-to-end RAG pipeline (daily use) |

---

## One-Time Setup (Index Building)

**Step 1** — `python 01_load_dataset.py`
- Scans 4 source folders (Advert_Gallery, OnlineAds, Epaper1, Epaper2)
- Extracts category/subcategory/brand from JSON annotations
- Outputs: `processed/metadata/madverse_metadata.csv` (61,576 rows, 492 brands)

**Step 2** — `python 02_generate_embeddings.py`
- Loads all 61,576 images through CLIP (`openai/clip-vit-base-patch32`)
- Generates L2-normalized 512-D embeddings per image
- Outputs: `embeddings/image_embeddings.pkl` (294 MB)

**Step 3** — `python 03_build_faiss_index.py`
- Loads embeddings, builds FAISS `IndexFlatL2(512)` index
- Outputs: `embeddings/faiss_indexes/madverse_index.faiss` + `id_to_metadata.pkl`

---

## Daily Usage — Ad Generation

```
python 09_pamphlet_pipeline.py "nike running shoes"
```

### Stage 0 — Brand Matching
- Fuzzy-matches query text against 492 known brands
- Identifies target brand (e.g., "Nike") for filtered retrieval

### Stage 1 — RAG Retrieval (CLIP + FAISS)
- Encodes query text into a 512-D CLIP embedding
- Builds a brand-filtered sub-index from FAISS (only target brand vectors)
- Retrieves top 5 most similar ad images from the dataset
- Extracts dominant colors via KMeans clustering

### Stage 2 — Ad Analysis (Gemini Vision RAG)
- Sends the 5 retrieved ad images to Gemini 2.0 Flash Vision
- Gemini extracts: tagline, product features, visual style, mood, color palette
- Gemini generates a detailed diffusion prompt for new ad creation
- Fallback chain: Gemini Vision → CLIP Descriptor RAG → Brand Knowledge Base

### Stage 3 — Image Generation (FLUX.1)
- Sends the LLM-crafted diffusion prompt to FLUX.1 model
- Routes: Pollinations.ai (free) → HuggingFace Inference API
- Generates a new product image based on the retrieved ad context

### Stage 4 — Pamphlet Compositing
- AI-driven 3x3 grid layout analysis for optimal text/logo placement
- Fetches brand logo from web (Clearbit + Google Favicon)
- Composes a 1080x1080 professional pamphlet with:
  - Full-bleed gradient background (brand colors)
  - Hero product image with rounded corners + shadow
  - Bold brand name (Impact font, multi-layer outline)
  - Tagline, feature highlights, CTA button
  - Reference thumbnails from retrieved dataset ads

**Output**: Final pamphlet saved to `outputs/` folder as PNG

---

## Architecture Flow

```
Query Text ──→ CLIP Embedding ──→ FAISS Search ──→ Top 5 Real Ads
                                                        │
                                                        ▼
                                                  Gemini Vision
                                                  analyzes them
                                                        │
                                                        ▼
                                              Text descriptions +
                                              diffusion prompt
                                                        │
                                                        ▼
                                                FLUX.1 generates
                                                  new ad image
                                                        │
                                                        ▼
                                              PamphletComposer
                                              creates final ad
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for CLIP embedding generation)
- API keys: Google Gemini, HuggingFace, Anthropic (optional)
- MAdVerse dataset (61,576 ad images with JSON annotations)
