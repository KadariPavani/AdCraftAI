# MAdVerse - Weekly Documentation

**Project Name:** MAdVerse (Marketing Advertisement Universe)
**Date:** 25 February 2026
**Branch:** develop
**Version:** 2.1.0

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset: What We Have](#2-dataset-what-we-have)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Runtime Ad Generation Pipeline](#4-runtime-ad-generation-pipeline)
5. [Known vs Unknown Brands/Products/Categories: How the Pipeline Behaves](#5-known-vs-unknown-brandsproductscategories-how-the-pipeline-behaves)
   - [5.1-5.8 Brand Scenarios (Known, Collision, Unknown, Override)](#51-case-a-known-brand-eg-nike-running-shoes-ad)
   - [5.9-5.12 Category Scenarios (Known, Unknown, Novel, Gap Analysis)](#59-known-vs-unknown-categories-how-categorization-works)
6. [Smart Prompt Parser](#6-smart-prompt-parser)
7. [Module-by-Module Breakdown](#7-module-by-module-breakdown)
8. [Technologies and Libraries Used](#8-technologies-and-libraries-used)
9. [API Endpoints](#9-api-endpoints)
10. [What From the Dataset Is Actually Used and Where](#10-what-from-the-dataset-is-actually-used-and-where)
11. [Dataset Enhancement (Self-Improving Loop)](#11-dataset-enhancement-self-improving-loop)
12. [File Structure](#12-file-structure)
13. [How to Run](#13-how-to-run)
14. [Pros and Cons](#14-pros-and-cons)

---

## 1. Project Overview

MAdVerse is a **Retrieval-Augmented Generation (RAG)** pipeline that generates professional advertisement creatives. Instead of using hardcoded templates, every design decision (colors, features, visual style, layout theme, image prompt, and ad copy) is derived from **real advertisement images** in the dataset combined with **AI-generated text**.

**Core idea:** Given a user query like *"Nike running shoes ad"*, the system:
1. Finds the brand in the dataset
2. Retrieves the most visually similar real ads for that brand
3. Extracts colors, style, and mood from those real ads
4. Generates all marketing text via AI
5. Generates a product image via AI
6. Composes a final 1080x1080 ad creative
7. Feeds the generated ad back into the dataset for future use

---

## 2. Dataset: What We Have

### 2.1 Raw Data Location

All images are stored under `data/images/` in **4 source collections**:

| Folder Path | Source Key | Image Count | Description |
|---|---|---|---|
| `data/images/Advert_Gallery/NewsPaperAds/Advert_Gallery/` | `adgal` | ~1,951 | English newspaper ads organized by **brand** (167 brand subfolders like Nike/, Amul/, Coca_Cola/) |
| `data/images/Epaper1/Epaper1/` | `epaper1` | ~20,564 | Regional e-paper ads organized by **language** (8 folders: Assamese, Bengali, Gujarati, Hindi, Marathi, Odia, Punjabi, Urdu) |
| `data/images/Epaper2/Epaper2/` | `epaper2` | ~16,683 | Regional e-paper ads organized by **language** (10 folders: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Tamil, Telugu, Urdu) |
| `data/images/OnlineAds/OnlineAds/` | `web` | ~22,397 | Web-scraped ads organized by **category** (11 folders: baby_products, body_wear, cosmetics, drinks, electronics, financial_institutions, food, home_essentials, sports, travel, vehicles) |

**Total: ~61,626 image files**

### 2.2 Annotations

JSON annotation files are stored in `data/annotations/`:
- `adgal_annot_j.json` - Advert Gallery annotations
- `web_annot_j.json` - Online Ads annotations
- `epaper1_annotation.json` - Epaper1 annotations
- `epaper2_annotation.json` - Epaper2 annotations

Each annotation entry contains:
```json
{
  "img_path": "path/to/image.jpg",
  "hier_annot": ["category", "subcategory", "brand"],
  "language": "english",
  "ad_type": "product"
}
```

### 2.3 Processed Metadata

After running `DatasetLoad.py`, all annotations are merged into a single CSV:
- **File:** `processed/metadata/madverse_metadata.csv`
- **Rows:** 61,595 (one per image)
- **Columns:** `source`, `source_folder`, `image_id`, `image_filename`, `category`, `subcategory`, `brand`, `language`, `ad_type`, `original_path`, `image_path`

### 2.4 Dataset Statistics

| Metric | Value |
|---|---|
| Total images indexed | 61,576 |
| Unique brands | 495 |
| Unique categories | 12 |
| Unique subcategories | 52 |
| Embedding dimension | 512 |
| FAISS index type | IndexFlatL2 (exact search) |

**Categories:** baby_products, body_wear, cosmetics, drinks, electronics, financial_institutions, food, home_essentials, sports, travel, unknown, vehicles

---

## 3. Data Preprocessing Pipeline

This is a **one-time setup** process. Three scripts run sequentially to transform raw images into a searchable vector index.

### Step 1: DatasetLoad.py - Scan Images to Metadata CSV

**What it does:**
1. Builds a file index by scanning all 4 image source folders (O(1) lookup by filename)
2. Loads all JSON annotation files from `data/annotations/`
3. Extracts `category`, `subcategory`, `brand`, `language`, `ad_type` from the hierarchical annotation (`hier_annot`) field
4. Links each annotation to its actual image file on disk using the pre-built index
5. Removes entries where the image file is missing

**Input:** `data/annotations/*.json` + `data/images/**/*`
**Output:** `processed/metadata/madverse_metadata.csv` (61,595 rows)

**Key logic in `_extract_row_from_madverse_item()`:**
```
hier_annot[0] -> category    (e.g., "sports")
hier_annot[1] -> subcategory (e.g., "sports_apparel")
hier_annot[2] -> brand       (e.g., "Adidas")
```

### Step 2: GenerateEmbeddings.py - Images to CLIP Embeddings

**What it does:**
1. Loads the metadata CSV from Step 1
2. Loads the CLIP model (`openai/clip-vit-base-patch32`) on GPU/CPU
3. For each image: opens it, processes through CLIP's vision encoder, L2-normalizes the output
4. Saves all embeddings as a Python dictionary: `{image_id: {embedding, image_path, source, category, brand, ...}}`
5. Includes a checkpoint system - saves progress every 1,000 images so interrupted runs can resume

**Input:** `processed/metadata/madverse_metadata.csv` + actual image files
**Output:** `embeddings/image_embeddings.pkl` (~281 MB)

**How embedding works:**
```
Image -> PIL.open() -> CLIPProcessor -> CLIPModel.get_image_features() -> L2 normalize -> 512-D float32 vector
```

**Requires:** `HF_TOKEN` environment variable (for HuggingFace model download)

### Step 3: BuildFAISS.py - Embeddings to FAISS Index

**What it does:**
1. Loads the embeddings pickle from Step 2
2. Stacks all embedding vectors into a numpy array (shape: 61,576 x 512)
3. Creates a FAISS `IndexFlatL2` index (exact L2 distance search, no approximation)
4. Adds all vectors to the index
5. Saves the index file and a separate metadata pickle mapping index IDs to image metadata

**Input:** `embeddings/image_embeddings.pkl`
**Output:**
- `embeddings/faiss_indexes/madverse_index.faiss` - The vector index
- `embeddings/faiss_indexes/id_to_metadata.pkl` - Maps integer ID to {image_path, brand, category, ...}
- `embeddings/faiss_indexes/index_stats.json` - Stats (61,576 vectors, 512 dims)

### Preprocessing Flow Diagram

```
data/annotations/*.json ──┐
                          ├──► DatasetLoad.py ──► madverse_metadata.csv
data/images/**/*.jpg ─────┘                              │
                                                         ▼
                                              GenerateEmbeddings.py
                                              (CLIP: openai/clip-vit-base-patch32)
                                                         │
                                                         ▼
                                              image_embeddings.pkl (281 MB)
                                                         │
                                                         ▼
                                                  BuildFAISS.py
                                                         │
                                              ┌──────────┼──────────┐
                                              ▼          ▼          ▼
                                        .faiss      .pkl       stats.json
                                       (index)   (metadata)   (61,576 vectors)
```

---

## 4. Runtime Ad Generation Pipeline

When a user submits a query, the pipeline executes **7 stages** sequentially:

### Stage 0: Brand Matching (`brands.py`)

**What happens:** The user query is fuzzy-matched against 495 known brand names.

**How it works:**
1. Tokenize query into words, filter out stop words ("ad", "create", "shoe", etc.)
2. For each word, fuzzy-match against all brand names using `difflib.SequenceMatcher` (cutoff: 0.5)
3. Also try word-pairs ("coca cola") and word-triples ("head & shoulders")
4. Return the best match with confidence score

**Input:** User query string (e.g., "Nike running shoes ad")
**Output:** `BrandMatch` with `matched_brand="Nike"`, `confidence=0.95`, `category="sports"`, `subcategory="sports_apparel"`, `image_count=N`

**Where dataset is used:** The brand index is built from `id_to_metadata` at startup - it scans all 61,576 metadata entries to find unique brands and their most common category/subcategory.

### Stage 1: FAISS Retrieval (`pipeline.py`)

**What happens:** Encode the query as a CLIP text embedding, search the FAISS index for the 5 most similar ad images.

**How it works:**
1. Encode query text through CLIP's text encoder -> 512-D vector
2. If a brand was matched in Stage 0:
   - Extract only that brand's vectors from the full index
   - Build a temporary sub-index with just those vectors
   - Search the sub-index (brand-filtered retrieval)
3. If no brand matched: search the full 61,576-vector index
4. Return top 5 results with similarity scores and metadata

**Input:** Query string + BrandMatch from Stage 0
**Output:** List of `RetrievedAd` objects with image_path, brand, category, subcategory, similarity score

**Where dataset is used:** The pre-computed CLIP embeddings stored in the FAISS index. No image files are opened here.

### Stage 2: Color Extraction (`colors.py`)

**What happens:** Extract dominant colors from the 5 retrieved reference ad images.

**How it works:**
1. Open each retrieved ad image file from disk via PIL
2. Resize each to 100x100 pixels
3. Combine all pixel data, sample 30,000 random pixels if too many
4. Run KMeans clustering (n_clusters=5, random_state=42)
5. Sort resulting colors by frequency (most dominant first)
6. Select accent color by scoring each for saturation and mid-range brightness
7. Generate a lightened background tint from the accent color

**Input:** Image paths of 5 retrieved ads
**Output:** List of 5 hex color strings + accent/secondary/background_tint

**Where dataset is used:** This is the **first stage that opens actual image files** from the dataset folders.

**Fallback:** If no image files can be opened, returns default palette: `["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"]`

### Stage 3: Content Generation (`content_gen.py` + `clip_extract.py`)

**What happens:** Two parallel processes:

**A) AI Text Generation (`content_gen.py`):**
1. Call Pollinations.ai text API with brand, category, and query
2. System prompt instructs it to return JSON with: headline, tagline, features (4-6), CTA text, product title, product description, Instagram caption, WhatsApp copy, hashtags
3. All text is generated fresh per request - no hardcoded marketing copy
4. Fallback: if JSON generation fails, tries plain text headline generation
5. Ultimate fallback: just the brand name

**B) CLIP-Based Prompt Generation (`clip_extract.py`):**
1. Open the 5 retrieved ad images and encode them through CLIP's vision encoder
2. Average the image features into a single 512-D vector
3. Rank candidate pools against this average:
   - **FEATURE_POOL** (48 candidates) -> top 6 features most visually relevant to the retrieved ads
   - **STYLE_POOL** (12 candidates) -> top 2 lighting/setting styles
   - **MOOD_POOL** (10 candidates) -> top 1 mood
   - **SUBJECT_POOL** (12 candidates) -> top 1 subject
4. Combine into a diffusion prompt: `"professional commercial advertisement photography for {brand}, {subject}, {styles}, {mood} mood, ultra realistic, 8k, sharp focus"`
5. Optionally, Pollinations AI enhances this prompt further

**C) Logo Fetching (`logo.py`):**
Fetches the brand logo from free sources with a 6-tier fallback:
1. Website scraping (apple-touch-icon meta tag)
2. Google faviconV2 API
3. icon.horse
4. DuckDuckGo icons
5. AI-generated logo via Pollinations (if enabled)
6. Text-based logo (brand initials on colored rectangle)

Background removal is applied to make the logo transparent.

**Where dataset is used:** The 5 retrieved ad images are opened again for CLIP encoding. Also, up to 3 are loaded as thumbnail images for the final pamphlet.

### Stage 4: Image Generation (`image_gen.py`)

**What happens:** Generate a product image using AI, with a 7-tier fallback chain.

**Fallback tiers (all free, no API keys required for tiers 1-3):**

| Tier | Service | Model | Requires |
|---|---|---|---|
| 1 | Pollinations.ai | FLUX | Nothing (free) |
| 2 | Pollinations.ai | Turbo | Nothing (free) |
| 3 | Pollinations.ai | flux-realism | Nothing (free) |
| 4 | HuggingFace Inference | FLUX.1-schnell | HF_TOKEN + credits |
| 5 | HuggingFace Inference | SDXL | HF_TOKEN (free model) |
| 6 | HuggingFace Inference | Stable Diffusion 2.1 | HF_TOKEN (free model) |
| 7 | Local PIL | Gradient fallback | Nothing |

**How it works:**
- Uses the diffusion prompt built in Stage 3 (derived from dataset reference images)
- Tries each tier in order; returns the first successful result
- Default image size: 1024x768
- The gradient fallback uses the colors extracted in Stage 2

**Where dataset is used:** Not directly. The prompt was informed by dataset images in Stage 3, but no dataset files are opened here.

### Stage 5: Multi-Language Translation (`content_gen.py`)

**What happens:** Translate all generated content into requested languages.

**Supported languages (25+):** English, Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Odia, Assamese, Urdu, Spanish, French, German, Portuguese, Arabic, Japanese, Korean, Chinese (Simplified), Russian, Italian, Dutch, Turkish

**How it works:**
1. English content is stored as the base
2. For each requested non-English language:
   - Translates: product_title, product_description, instagram_caption, whatsapp_copy, tagline
   - Uses `deep_translator.GoogleTranslator` (free, no API key)

**Where dataset is used:** Not used.

### Stage 6: Ad Composition (`designer.py`)

**What happens:** Compose the final 1080x1080 PNG ad creative.

**Theme selection (data-driven):**
The designer analyzes the retrieved reference ad images for:
- Average brightness
- Average saturation
- Edge complexity

Then weights probabilities for 6 layout themes:

| Theme | When Selected | Description |
|---|---|---|
| `minimal_clean` | Light, clean reference ads | Product hero top, wave divider, centered text bottom |
| `bold_hero` | Dark, high-contrast ads | Product top, dark bottom, bold uppercase headline |
| `premium_dark` | Low brightness, elegant ads | Gradient fade, centered elegant typography |
| `split_layout` | Moderate brightness/saturation | Product left, text right on accent background |
| `card_float` | Medium complexity ads | Product top, wave divider, accent text card |
| `gradient_mesh` | High saturation, vibrant ads | Vibrant mesh gradient below product image |

**Multi-script font support:**
- **Latin:** Segoe UI, Georgia, Impact, Times, Calibri
- **Indic (Hindi, Bengali, Tamil, Telugu, etc.):** Nirmala UI
- **CJK (Chinese, Japanese, Korean):** Microsoft YaHei
- **Arabic/Urdu:** Segoe UI

**Bilingual rendering:** When both English and a non-English language are requested, the ad displays text in both languages simultaneously (primary headline + secondary headline, etc.).

**What goes on the final ad:**
- Product image (from Stage 4)
- Brand logo (from Stage 3)
- Headline, tagline, features, CTA text (from Stage 3, translated in Stage 5)
- Colors: accent, secondary, background tint (from Stage 2)
- Thumbnail images from retrieved ads (from Stage 1)

**Output:** 1080x1080 PNG saved to `outputs/pamphlet_{timestamp}.png`

### Stage 7: Dataset Enhancement (`dataset_enhancer.py`)

**What happens:** The generated ad is fed back into the dataset so future queries can retrieve it.

(Detailed in [Section 9](#9-dataset-enhancement-self-improving-loop))

### Full Pipeline Flow

```
User Query: "Nike running shoes ad"
    │
    ▼
Stage 0: Brand Matching ─────────── metadata only (no files opened)
    │   Result: brand=Nike, category=sports, confidence=0.95
    ▼
Stage 1: FAISS Retrieval ─────────── pre-computed embeddings only
    │   Result: top 5 similar Nike ads from dataset
    ▼
Stage 2: Color Extraction ────────── *** OPENS dataset image files ***
    │   Result: accent=#e94560, secondary=#1a1a2e, bg=#f8f0f0
    ▼
Stage 3: Content + Prompt ────────── *** OPENS dataset image files ***
    │   Result: headline, tagline, features, CTA, diffusion prompt
    │   Also: logo fetched, thumbnails loaded
    ▼
Stage 4: Image Generation ────────── uses prompt (no dataset files)
    │   Result: 1024x768 product image
    ▼
Stage 5: Translation ─────────────── no dataset files
    │   Result: content in en, hi, ta, etc.
    ▼
Stage 6: Ad Composition ──────────── uses thumbnails from Stage 3
    │   Result: 1080x1080 pamphlet PNG
    ▼
Stage 7: Dataset Enhancement ─────── writes new image back into dataset
    │   Result: FAISS index updated, CSV updated
    ▼
Output: pamphlet PNG + product image + JSON result + all text content
```

---

## 5. Known vs Unknown Brands/Products/Categories: How the Pipeline Behaves

The pipeline behaves very differently depending on whether the brand, product, or category in the user's query exists in the dataset or not. This section traces what each stage does in all cases, the problems that arise, and the current workarounds. Sections 5.1-5.8 cover **brand** scenarios, sections 5.9-5.12 cover **category** scenarios.

### 5.1 Case A: Known Brand (e.g., "Nike running shoes ad")

"Nike" exists in the dataset with ~N images under the brand `Nike`.

| Stage | What Happens | Result |
|---|---|---|
| **Stage 0: Brand Matching** | "nike" fuzzy-matches `Nike` in brand index with high confidence (~0.95). Returns `category="sports"`, `subcategory="sports_apparel"`. | Correct brand, correct category |
| **Stage 1: FAISS Retrieval** | Builds a **sub-index** from only Nike's FAISS vectors. Searches within that subset. Returns top 5 Nike ads most semantically similar to "running shoes". | Correct brand-filtered results |
| **Stage 2: Color Extraction** | Opens the 5 retrieved Nike ad images. KMeans extracts Nike's dominant colors (likely black, white, red, orange). | Brand-appropriate colors |
| **Stage 3: Content Generation** | `brand = "Nike"`, `category = "sports"`. Pollinations AI generates text specifically for Nike running shoes. CLIP ranks styles/moods against Nike ad images. | Correct brand context for all text |
| **Stage 3: Logo** | `Nike` found in `BRAND_DOMAINS` dict -> `nike.com`. Fetches the Nike swoosh logo from the website. | Correct brand logo |
| **Stage 4: Image Generation** | Diffusion prompt: "professional commercial advertisement photography for Nike, athlete in dynamic action pose, ..." | Brand-relevant image |
| **Stage 5: Translation** | Translates Nike shoe ad text into requested languages. | Correct |
| **Stage 6: Composition** | Theme selected based on Nike ad visual characteristics. Colors, logo, text, thumbnails all match Nike. | Cohesive Nike ad |
| **Stage 7: Enhancement** | Saves generated pamphlet into Nike's existing folder in the dataset. Adds to FAISS index under brand "Nike". | Correct dataset placement |

**Everything works as designed.** The brand acts as the anchor for the entire pipeline.

### 5.2 Case B: Unknown Brand That Collides With a Known Brand (e.g., "Apple fruit ad")

"Apple" exists in the dataset as `Apple_mobiles` (electronics). The user means apple the fruit.

| Stage | What Happens | Problem |
|---|---|---|
| **Stage 0: Brand Matching** | "apple" fuzzy-matches `Apple_mobiles` with high confidence (~0.8). Returns `category="electronics"`, `subcategory="mobiles"`. | **False positive** - fruit "apple" is confused with Apple the tech company |
| **Stage 1: FAISS Retrieval** | Builds sub-index from Apple_mobiles images only. Returns top 5 iPhone/MacBook ad images. | **Completely wrong** reference images |
| **Stage 2: Color Extraction** | Extracts colors from Apple electronics ads (white, silver, black, space gray). | **Wrong colors** - should be red/green for fruit |
| **Stage 3: Content Generation** | `brand = "Apple_mobiles"`, `category = "electronics"`. AI generates text about Apple electronics, NOT apple fruit. CLIP ranks styles against iPhone images. | **Wrong product context** - entire text is about phones |
| **Stage 3: Logo** | Fetches apple.com logo (Apple Inc. bitten apple logo). | **Wrong logo** |
| **Stage 4: Image Generation** | Prompt built around electronics photography. | **Wrong image** generated |
| **Stage 6: Composition** | Final ad looks like an Apple iPhone/MacBook ad with tech-style colors and electronic product imagery. | **Completely wrong final output** |
| **Stage 7: Enhancement** | Saves a fruit-intended ad into the `Apple_mobiles` dataset folder, further polluting it. | **Dataset pollution** |

**Root cause:** The fuzzy matcher (`difflib`, cutoff=0.5) has no semantic understanding. It only compares character sequences. "apple" (fruit) and "apple" (brand) are identical strings, so the brand always wins.

**There is no current fix for this in the codebase.** The system has no concept of word-sense disambiguation.

### 5.3 Case C: Completely Unknown Brand, No Collision (e.g., "XOXO brand shoes ad")

"XOXO" does not exist in the dataset and does not fuzzy-match any of the 495 known brands.

| Stage | What Happens | Details |
|---|---|---|
| **Stage 0: Brand Matching** | "xoxo" does not match any brand (no close match above 0.5 cutoff). Returns **empty** `BrandMatch()` with `matched_brand=None`, `category=""`, `subcategory=""`. | Correct - no false positive |
| **Stage 1: FAISS Retrieval** | Since `matched_brand` is None, **searches the FULL 61,576-vector index** with no brand filtering. CLIP encodes the text "XOXO brand shoes ad" and retrieves the 5 most semantically similar ads from ANY brand. Likely returns shoe/sports ads from Nike, Adidas, Puma, etc. | Reasonable fallback - retrieves visually relevant ads |
| **Stage 2: Color Extraction** | Opens the 5 retrieved ads (e.g., Nike, Adidas shoe ads). Extracts their dominant colors. | Colors come from other shoe brands, not XOXO-specific |
| **Stage 3: Content Generation** | **This is where the key problem occurs.** At `pipeline.py:247-249`: | |
| | `brand = brand_match.matched_brand or (retrieved_ads[0].brand if retrieved_ads else "Product")` | Since `matched_brand` is None, falls back to `retrieved_ads[0].brand` which could be "Nike" |
| | `category = brand_match.category or (retrieved_ads[0].category if retrieved_ads else "product")` | Category becomes "sports" (from the retrieved Nike ad) |
| | AI now generates text for "Nike" instead of "XOXO". The user's brand name "XOXO" is **lost entirely**. | **Wrong brand name used for all text generation** |
| **Stage 3: Logo** | Fetches logo for the retrieved brand (e.g., Nike swoosh), NOT XOXO. | **Wrong logo** |
| **Stage 4: Image Generation** | Diffusion prompt says "professional commercial advertisement photography for Nike..." | **Wrong brand in prompt** |
| **Stage 6: Composition** | Final ad headline, features, CTA, and logo are all for Nike (or whichever brand's ad was retrieved first). | **Final ad is for the wrong brand** |
| **Stage 7: Enhancement** | Saves into the retrieved brand's folder (e.g., Nike/). No "XOXO" folder is created. | **No new brand registered** |

**The fundamental issue at `pipeline.py:247`:**
```python
brand = brand_match.matched_brand or (retrieved_ads[0].brand if retrieved_ads else "Product")
```
When brand matching fails, the system does NOT preserve the user's original brand name from the query. Instead, it uses whatever brand the first retrieved ad belongs to.

### 5.4 Case D: Unknown Brand WITH `brand_override` (e.g., "XOXO shoes" + brand="XOXO")

The user explicitly passes the brand name through the API's `brand` form field. This is the **only path that correctly handles new brands**.

| Stage | What Happens | Details |
|---|---|---|
| **Stage 0: Brand Matching** | Same as Case C - no match found. Returns empty `BrandMatch()`. | No false match |
| **Stage 1: FAISS Retrieval** | Same as Case C - full-index search. Returns top 5 shoe ads from various brands. | Semantic fallback works |
| **Stage 2: Color Extraction** | Same as Case C - extracts colors from retrieved ads. | Colors from similar product ads (reasonable) |
| **Stage 3: Content Generation** | At `pipeline.py:244-245`: `if brand_override: brand = brand_override.replace(" ", "_")` | **`brand = "XOXO"` - correct!** |
| | AI generates text specifically for XOXO brand shoes. | Correct brand context |
| **Stage 3: Logo** | `_get_domain("XOXO")` -> heuristic builds `"xoxo.com"` (`logo.py:89-95`). Tries fetching logo from that domain. If the website doesn't exist or has no icon, falls through all 4 web sources and finally generates a **text-based logo** with the initials "XO" on a colored rounded rectangle. | Reasonable fallback logo |
| **Stage 4: Image Generation** | Diffusion prompt says "professional commercial advertisement photography for XOXO..." | Correct brand in prompt |
| **Stage 6: Composition** | Final ad uses XOXO name, AI-generated XOXO-specific text, text logo, colors from similar shoe ads. | **Correct output for new brand** |
| **Stage 7: Enhancement** | `_find_brand_folder("XOXO")` returns None (brand doesn't exist). `_save_to_dataset()` creates **new folder** at `data/images/Advert_Gallery/NewsPaperAds/Advert_Gallery/XOXO/`. Registers "XOXO" as a new brand in FAISS index and brand_matcher. | **New brand folder created, brand registered** |

**After the first generation with `brand_override`:** XOXO now exists in the dataset. Future queries for "XOXO" will fuzzy-match and work like Case A (known brand). The system bootstraps new brands through the `brand_override` mechanism.

### 5.5 Stage-by-Stage Comparison Table

| Stage | Known Brand (Nike) | Unknown Brand, No Override (XOXO) | Unknown Brand, With Override (XOXO) | Collision (Apple fruit) |
|---|---|---|---|---|
| **Brand Match** | `Nike`, confidence=0.95 | None | None (override used instead) | `Apple_mobiles`, confidence=0.8 (wrong) |
| **FAISS Search** | Nike sub-index only | Full 61K index | Full 61K index | Apple_mobiles sub-index (wrong) |
| **Retrieved Ads** | Nike ads | Shoe ads from any brand | Shoe ads from any brand | iPhone/MacBook ads (wrong) |
| **Colors** | Nike's brand colors | Mixed shoe brand colors | Mixed shoe brand colors | Apple electronics colors (wrong) |
| **Brand Name Used** | "Nike" | **"Nike"** (from 1st retrieved ad - wrong) | **"XOXO"** (from override - correct) | "Apple_mobiles" (wrong context) |
| **Category Used** | "sports" (from brand index) | "sports" (from 1st retrieved ad) | "sports" (from 1st retrieved ad) | "electronics" (wrong) |
| **AI Text** | About Nike shoes | About Nike shoes (wrong brand) | About XOXO shoes (correct) | About Apple phones (wrong) |
| **Logo** | Nike swoosh (correct) | Nike swoosh (wrong) | "XO" text logo (reasonable) | Apple Inc. logo (wrong) |
| **Image Prompt** | "...for Nike..." | "...for Nike..." (wrong) | "...for XOXO..." (correct) | "...for Apple_mobiles..." (wrong) |
| **Final Ad** | Correct Nike ad | Wrong - looks like Nike ad | Correct XOXO ad | Wrong - looks like iPhone ad |
| **Dataset Save** | Into Nike/ folder | Into Nike/ folder (wrong) | Creates XOXO/ folder (correct) | Into Apple_mobiles/ (pollutes) |

### 5.6 How the brand_override Mechanism Works

The `brand_override` is the **only reliable way** to generate ads for brands not in the dataset.

**API call flow:**

1. User sends POST to `/api/generate` with `brand="XOXO"` form field
2. `main.py:209` passes `brand_override="XOXO"` to `pipeline.generate()`
3. `pipeline.py:244-245` checks:
   ```python
   if brand_override:
       brand = brand_override.replace(" ", "_")  # brand = "XOXO"
   else:
       brand = brand_match.matched_brand or (retrieved_ads[0].brand ...)
   ```
4. From this point, `brand = "XOXO"` is used for ALL downstream stages

**What this fixes:**
- AI text generation gets the correct brand name
- Logo fetching tries the correct domain
- Image prompt references the correct brand
- Dataset enhancement creates a new folder for the brand
- Brand matcher registers the new brand for future queries

**What this does NOT fix:**
- Category and subcategory still fall back to the first retrieved ad's values (no independent classification)
- Colors still come from whatever ads were retrieved (not brand-specific)
- The retrieved reference ads are still from other brands (since the new brand has no images yet)

### 5.7 What Happens After a New Brand's First Generation

After XOXO's first ad is generated (with `brand_override`):

1. **FAISS index** now contains XOXO's pamphlet embedding (1 vector)
2. **id_to_metadata** has an entry with `brand="XOXO"`, `category="sports"`, `subcategory="sports_apparel"`
3. **brand_matcher.brand_index** has `XOXO: {indices: [new_id], category: "sports", count: 1}`
4. **Dataset folder** `data/images/.../XOXO/` contains `XOXO_gen_{timestamp}.png`

**On the second query for "XOXO shoes ad":**
- Stage 0: "xoxo" now matches `XOXO` in brand_matcher (exact match, confidence=1.0)
- Stage 1: Builds sub-index from XOXO's 1 image, retrieves it
- Stage 2: Extracts colors from the previously generated pamphlet
- Stage 3: Uses "XOXO" as brand name (from brand_match this time)
- Stage 7: Adds second image to XOXO/ folder, now has 2 images

**The brand bootstraps itself.** Each generation adds to the dataset, making future generations more brand-consistent. After several generations, XOXO will have its own color palette, its own reference images, and its own visual identity in the system.

### 5.8 Known Limitations for Unknown Products/Brands

| Limitation | Description | Impact |
|---|---|---|
| **No word-sense disambiguation** | "Apple" (fruit) vs "Apple" (brand) cannot be distinguished. The fuzzy matcher only compares character sequences, not meaning. | False positives for common words that are also brand names |
| **Brand fallback uses retrieved ad's brand** | When matching fails and no `brand_override` is given, the user's intended brand name is lost. The first retrieved ad's brand is used instead. | Wrong brand name on the final ad |
| **No independent product/item classification** | Category is always inherited from the brand (or the retrieved ad's brand). There is no classifier that looks at the query and independently determines "this is about footwear" or "this is about fruit". | A brand that spans multiple categories (Samsung: phones, TVs) always gets its majority category |
| **brand_override is the only correct path for new brands** | Without explicit brand input, the system cannot correctly generate ads for brands not in the dataset. | Users must know to provide the brand name separately |
| **First generation for a new brand uses other brands' references** | Since the new brand has no images in the dataset yet, FAISS retrieves ads from other similar brands. Colors, styles, and thumbnails are from those other brands. | The first ad for a new brand may look visually similar to competitors |
| **Category assigned to new brand is inherited** | When XOXO is first added via dataset enhancement, it gets the category from the first retrieved ad (e.g., "sports" from Nike). This may not be correct for XOXO. | Permanent incorrect category assignment if the first retrieval was wrong |

### 5.9 Known vs Unknown Categories: How Categorization Works

The system has **12 known categories** in the dataset: baby_products, body_wear, cosmetics, drinks, electronics, financial_institutions, food, home_essentials, sports, travel, unknown, vehicles. However, there is **no independent category classifier** anywhere in the pipeline. Category is always derived indirectly.

#### How Category Is Determined (Code Path)

The category for any generation is resolved at `pipeline.py:248-249`:

```python
category = brand_match.category or (retrieved_ads[0].category if retrieved_ads else "product")
subcategory = brand_match.subcategory or (retrieved_ads[0].subcategory if retrieved_ads else "")
```

This means there are only **two sources** for category:

| Source | When Used | How It Was Built |
|---|---|---|
| `brand_match.category` | When a brand is successfully matched | At startup, `BrandMatcher.__init__()` scans all metadata entries for each brand and picks the **most common** category using `Counter.most_common(1)` (`brands.py:119`) |
| `retrieved_ads[0].category` | When brand matching fails (fallback) | Whatever category the first FAISS-retrieved ad had in its metadata |

**There is no third option.** The system never analyzes the query text to independently classify the product.

#### Case A: Known Category Through Known Brand

**Query:** "Samsung mobiles ad"

```
Step 1: "samsung" matches brand "Samsung_mobiles" (confidence ~0.9)
Step 2: brand_match.category = "electronics" (from brand index, most common for Samsung)
Step 3: brand_match.subcategory = "mobiles"
Step 4: category = "electronics" ✓ (used for AI text, dataset enhancement, metadata)
```

**Result:** Category is correct because Samsung is a known brand and "electronics" is its majority category.

#### Case B: Known Category Through FAISS Retrieval (No Brand Match)

**Query:** "new smartphone ad"

```
Step 1: No word matches any brand (all are stop words or too generic)
Step 2: brand_match = empty (matched_brand=None, category="")
Step 3: CLIP encodes "new smartphone ad" → searches full 61K index
Step 4: Top retrieved ad might be Samsung_mobiles or Apple_mobiles
Step 5: category = retrieved_ads[0].category = "electronics" (from that ad's metadata)
Step 6: brand = retrieved_ads[0].brand = "Samsung_mobiles" (wrong — user didn't say Samsung)
```

**Result:** Category happens to be correct ("electronics") because the CLIP semantic search retrieved a relevant phone ad. But the brand is wrong.

#### Case C: Wrong Category Due to Brand Collision

**Query:** "Apple fruit juice ad"

```
Step 1: "apple" matches brand "Apple_mobiles" (confidence ~0.8)
Step 2: brand_match.category = "electronics" ✗ (should be "food" or "drinks")
Step 3: brand_match.subcategory = "mobiles" ✗ (should be "juice" or "beverages")
Step 4: category = "electronics" (WRONG — used in AI text prompt, dataset storage)
```

**AI text prompt sends to Pollinations:**
```
Brand: Apple mobiles
Category: electronics
Product: fruit juice
User request: Apple fruit juice ad
```

The AI receives contradictory signals — "electronics" category but "fruit juice" product. Output quality depends on whether the AI trusts the category or the product description.

#### Case D: Unknown Category (Product Not in Any of the 12 Categories)

**Query:** "FreshAir air purifier ad"

```
Step 1: "freshair" does not match any brand → brand_match = empty
Step 2: CLIP encodes "FreshAir air purifier ad" → semantic search
Step 3: FAISS retrieves ads that are visually/semantically closest
        Could be: electronics (if "air purifier" ~ "appliance" in CLIP space)
        Or: home_essentials (if it matches home products)
        Or: vehicles (if CLIP associates "air" with car ads)
Step 4: category = retrieved_ads[0].category (whatever the top result had)
```

**Result:** Category is essentially random — it depends on which of the 61,576 ads CLIP considers most similar to "FreshAir air purifier". Since "air purifier" is not a standard category in the dataset, the assigned category could be anything.

#### Case E: Completely Novel Product Category

**Query:** "SpaceX rocket tourism ad"

```
Step 1: "spacex" does not match any brand
Step 2: CLIP encodes "SpaceX rocket tourism ad"
Step 3: FAISS might retrieve:
        - travel ads (because "tourism")
        - vehicle ads (because "rocket" ~ vehicle in CLIP space)
        - electronics ads (if "SpaceX" ~ tech in CLIP space)
Step 4: category = whatever the top ad's category was (e.g., "travel")
```

**Result:** No category like "aerospace" or "space tourism" exists in the 12-category taxonomy. The system maps it to the nearest existing category via FAISS retrieval.

### 5.10 Where Category Is Used Downstream

Once the category is determined, it flows through these stages:

| Stage | How Category Is Used | Impact of Wrong Category |
|---|---|---|
| **AI Text Generation** (`content_gen.py:146`) | Sent to Pollinations AI in the prompt: `"Category: {category}"`. The AI uses this to decide what kind of features, tagline, and CTA to generate. | AI may generate irrelevant features (e.g., "5G connectivity" for a food product if category="electronics") |
| **CLIP Feature Ranking** (`clip_extract.py:148-153`) | Category is passed to `extract_features()` but **not actually used** — features are ranked purely by CLIP visual similarity to reference images, ignoring the category string. | No impact — CLIP ranking is category-agnostic |
| **Diffusion Prompt** (`clip_extract.py:155`) | Category is passed to `generate_diffusion_prompt()` but **not used in the prompt text**. The prompt is built from CLIP-ranked styles/moods/subjects. | No impact — prompt is image-driven, not category-driven |
| **Theme Selection** (`designer.py:720`) | Category is **NOT used**. Theme is selected by analyzing reference ad brightness/saturation/complexity. | No impact — theme is visual-analysis-driven |
| **Dataset Enhancement** (`dataset_enhancer.py:113-114`) | Category is stored in the FAISS metadata for the new image. Future retrievals for this brand will inherit this category. | Wrong category persists permanently in the dataset |
| **Metadata CSV** (`dataset_enhancer.py:260`) | Category is written as a column in `madverse_metadata.csv`. | Wrong label in the CSV |
| **Brand Matcher** (`dataset_enhancer.py:241`) | For new brands, the category is stored in `brand_index[brand]["category"]`. All future brand matches will return this category. | Permanent wrong category for this brand |

**Key insight:** Category primarily affects **AI text generation** and **metadata storage**. It does NOT affect colors, theme selection, CLIP feature ranking, or image generation prompts — those are all driven by visual analysis of retrieved images.

### 5.11 Category Handling: Summary Comparison

```
Scenario                          │ Category Source           │ Correct?  │ Why
──────────────────────────────────┼───────────────────────────┼───────────┼────────────────────────
Known brand, single category      │ Brand index (most common) │ ✓ Yes     │ Nike → "sports" is right
(e.g., "Nike shoes")              │                           │           │
                                  │                           │           │
Known brand, multi-category       │ Brand index (most common) │ ~ Partial │ Samsung → "electronics"
(e.g., "Samsung washing machine") │                           │           │ even though appliances ≠ mobiles
                                  │                           │           │
Brand collision with diff meaning │ Brand index (wrong brand) │ ✗ Wrong   │ "Apple fruit" → "electronics"
(e.g., "Apple fruit juice")       │                           │           │ because Apple_mobiles matched
                                  │                           │           │
No brand match, clear product     │ First retrieved ad        │ ~ Maybe   │ "smartphone ad" retrieves
(e.g., "new smartphone ad")       │                           │           │ phone ads → "electronics" likely
                                  │                           │           │
No brand match, novel product     │ First retrieved ad        │ ~ Random  │ "air purifier ad" could get
(e.g., "air purifier ad")         │                           │           │ "electronics" or "home_essentials"
                                  │                           │           │
Completely novel category          │ First retrieved ad        │ ✗ Wrong   │ "rocket tourism" forced into
(e.g., "SpaceX rocket tourism")   │                           │           │ nearest of 12 existing categories
                                  │                           │           │
New brand with brand_override     │ First retrieved ad        │ ~ Maybe   │ Override fixes brand name but
(e.g., brand="XOXO")             │                           │           │ category still from retrieved ad
```

### 5.12 The Category Gap: What the System Lacks

The current system has **no independent category/product classifier**. This means:

1. **No query analysis for product type** — The words "shoes", "juice", "phone", "car" in the query are never parsed to determine category. The query is only used for brand matching (word-level fuzzy match) and FAISS retrieval (full-sentence CLIP embedding).

2. **No CLIP-based category classification** — CLIP could classify the query text against the 12 category labels (e.g., compute similarity of "Apple fruit juice" against ["food", "drinks", "electronics", ...]) to independently determine category. This is not implemented.

3. **No user-provided category override** — Unlike `brand_override`, there is no `category_override` parameter in the API. Users cannot explicitly specify that their query is about "food" vs "electronics".

4. **Fixed 12-category taxonomy** — The dataset has exactly 12 categories. Products that don't fit (SaaS, education, real estate, aerospace, etc.) are forced into the nearest existing category with no way to expand the taxonomy at runtime.

5. **Category is permanently assigned** — Once a brand's category is set (either from the original dataset or from dataset enhancement), it cannot be changed without manually editing the metadata CSV and rebuilding the index.

---

## 6. Smart Prompt Parser

### Overview

The Smart Prompt Parser (`app/smart_prompt.py`) extracts structured product catalog data from free-text descriptions. It powers the "Analyze & Continue" step in the UI wizard, enabling users to type a natural language product description and have it automatically parsed into structured fields (brand, product name, type, category, price, size, material, color, features, etc.).

**Performance:** <50ms for all prompts (zero API calls in default mode). Optional AI-enhanced mode adds ~3-8s via Pollinations.

### Architecture

```
User Prompt: "Nike Air Max 90 running shoes for men, black, Rs 12,995, UK 7-12"
    |
    v
1. Scene Masking ---------> Extract scene description first, mask from product detection
    |                       ("show an Indian bride in red silk saree" -> masked)
    v
2. Brand Detection -------> Known brands (150+, longest-first) -> Dataset fuzzy match -> CamelCase heuristic
    |                       Result: brand="Nike"
    v
3. Product Type ----------> 150+ types globally sorted by length, word-boundary matching
    |                       "water purifier" beats "water", "running shoes" beats "shoes"
    v
4. Field Extraction ------> Regex patterns for price, size, color, material, audience, etc.
    |                       All matching uses \b word boundaries to prevent false positives
    v
5. Category Inference ----> Weighted keyword scoring + brand-category knowledge map
    |                       "smartphone" (3pts) beats "phone" (2pts)
    v
6. Product Name ----------> Quoted text -> explicit patterns -> brand+model -> fallback to type
    |
    v
7. Completeness Check ----> Compare extracted fields against category-specific required fields
    |
    v
Output: { extracted fields, category, missing_required, completeness score, rich_prompt }
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Scene masking before product detection** | Prevents scene elements (e.g., "saree" in "bride wearing red silk saree") from being detected as the product type |
| **Word boundary matching everywhere** | Prevents substring false positives: "casio" inside "occasion", "hp" inside "shampoo", "lassi" inside "classic" |
| **Global longest-match product types** | Ensures "water purifier" (home_appliances) matches before "water" (beverages) regardless of category iteration order |
| **Brand-category knowledge map** | 150+ brands with default categories. Knowing "Nike" = footwear helps even when no product type keyword is found |
| **Single-word brand heuristic for unknowns** | Takes only the first CamelCase word as brand ("TerraMotors EcoRide" -> brand=TerraMotors, name=EcoRide) to avoid absorbing model names |

### Supported Categories & Required Fields

| Category | Required Fields |
|----------|----------------|
| Footwear | brand, product_name, product_type, target_audience, size_range |
| Electronics | brand, product_name, product_type, key_features |
| Clothing & Apparel | brand, product_name, product_type, target_audience, size_range |
| Beverages | brand, product_name, product_type, pack_size |
| Jewelry & Watches | brand, product_name, product_type, material |
| Automotive | brand, product_name, product_type, key_features |
| Food & Snacks | brand, product_name, product_type, pack_size |
| Personal Care & Beauty | brand, product_name, product_type, target_audience |
| Home Appliances | brand, product_name, product_type, key_features |
| General Product | brand, product_name, product_type |

### Example Prompts

**Known Brands (in dataset):**

| Prompt | Extracted |
|--------|-----------|
| `Nike Air Max 90 men's running shoes, sizes UK 7-12, black/white/infrared, mesh and leather, Rs 12,995` | Brand=Nike, Name=Air Max 90 running shoes, Type=Running Shoes, Category=footwear, Audience=Men, Size=UK 7-12, Color=Black, White, Material=Mesh, Leather, Price=Rs 12,995 |
| `Samsung Galaxy S24 Ultra smartphone, 200MP camera, Snapdragon 8 Gen 3, 12GB RAM, 5000mAh battery, titanium frame, Rs 1,29,999` | Brand=Samsung, Name=Galaxy S24 Ultra smartphone, Type=Smartphone, Category=electronics, Features=200MP camera, 5000mAh battery, 12GB RAM, Snapdragon 8 Gen 3 |
| `Kalyan Jewellers 22K gold bangles set for women, Rs 1,80,000, BIS hallmarked, sizes 2.4 to 2.8, show bride in saree at showroom` | Brand=Kalyan Jewellers, Type=Gold Bangles, Category=jewelry, Material=22K Gold, Scene=show bride in saree at showroom (masked from product detection) |

**New / Unknown Brands (no dataset entry):**

| Prompt | Extracted |
|--------|-----------|
| `StrideFlex AeroGlide running shoes for men, lightweight mesh, breathable cushioning, UK 7-12, black and neon green, Rs 5499` | Brand=StrideFlex, Name=AeroGlide running shoes, Type=Running Shoes, Category=footwear, Audience=Men |
| `NovaTech Pulse X1 wireless earbuds, 50hr battery, active noise cancellation, Bluetooth 5.4, IPX7 waterproof, Rs 3999` | Brand=NovaTech, Name=Pulse X1 wireless earbuds, Type=Wireless Earbuds, Category=electronics |
| `DermaCure vitamin C brightening face wash with niacinamide, for oily skin, paraben free, 150ml, Rs 449` | Brand=DermaCure, Name=vitamin C brightening face wash, Type=Face Wash, Category=personal_care |
| `VoltRider Storm 200 electric motorcycle, 150km range, 8kW motor, matte black, Rs 1,85,000, show rider on highway at sunset` | Brand=VoltRider, Name=Storm 200 electric motorcycle, Type=Motorcycle, Category=automotive |
| `RaniGold bridal choker necklace set 22k gold, uncut polki diamonds, BIS hallmarked, for women, wedding, 85 grams, Rs 3,50,000` | Brand=RaniGold, Name=bridal choker necklace set, Type=Necklace, Category=jewelry |
| `CrunchBox quinoa puffs 120g pack, tangy tomato flavor, baked not fried, high protein, gluten free, Rs 99` | Brand=CrunchBox, Name=quinoa puffs, Type=Chips, Category=food, Flavor=Tomato |

### API Endpoints

| Endpoint | Method | Parameters | Description |
|----------|--------|------------|-------------|
| `/api/parse-prompt` | POST | `prompt` (required), `use_ai` (optional, default "false") | Parse free-text into structured product data. Returns extracted fields, category, completeness score, missing required fields |
| `/api/validate-fields` | POST | `parsed_data` (JSON), `additional_fields` (JSON) | Merge user-provided fields into previously parsed data and recalculate completeness |
| `/api/categories` | GET | - | List all supported categories with their required and optional fields |

---

## 7. Module-by-Module Breakdown

### 6.1 `app/models.py` - Data Classes and Constants

Defines all shared data structures and path configuration.

**Path Configuration:**
```
BASE_DIR        = project root (MAdVerse/)
EMBEDDINGS_DIR  = BASE_DIR / embeddings
FAISS_DIR       = EMBEDDINGS_DIR / faiss_indexes
OUTPUT_DIR      = BASE_DIR / outputs
UPLOAD_DIR      = BASE_DIR / uploads
DB_DIR          = BASE_DIR / products_db
```

**Data Classes:**
- `RetrievedAd` - A single retrieved ad from FAISS (rank, similarity, distance, image_path, brand, category, subcategory, language, ad_type, source)
- `BrandMatch` - Result of brand matching (matched_brand, confidence, category, subcategory, image_count)
- `AdContent` - All content for composing an ad (brand_name, headline, tagline, features, colors, product_image, logo_image, thumbnails, diffusion_prompt, CTA, bilingual secondary text)
- `GenerationResult` - Complete pipeline output (query, timings, retrieved_ads, colors, content, pamphlet_path, product_image_path, translations, errors, dataset_paths)

**Constants:**
- `SUPPORTED_LANGUAGES` - 25 language codes with display names
- `DEFAULT_NEGATIVE_PROMPT` - Prevents artifacts in AI image generation (text, watermarks, blurry, extra fingers, etc.)

### 6.2 `app/brands.py` - Brand Matching

**`BRAND_TAGLINES` dictionary (57 brands):** Pre-defined taglines for known brands (Nike: "Just Do It", Adidas: "Impossible Is Nothing", etc.). These serve as a knowledge base.

**`BrandMatcher` class:**
- **Constructor:** Scans all 61,576 metadata entries. For each unique brand, stores:
  - List of FAISS index IDs belonging to that brand
  - Most common category (via `Counter.most_common(1)`)
  - Most common subcategory
  - Total image count
- **`match(query)`:** Fuzzy matching with 3 passes:
  1. Single-word matching (e.g., "nike" matches "Nike")
  2. Word-pair matching (e.g., "coca cola" matches "Coca_Cola")
  3. Word-triple matching (e.g., "head & shoulders" matches "Head_&_Shoulders_shampoo")
  - Uses `difflib.get_close_matches` with cutoff=0.5
  - Returns best match by `SequenceMatcher.ratio()`
- **`STOP_WORDS`:** Filters out common words like "ad", "create", "shoes", "water", etc. before matching

### 6.3 `app/colors.py` - Color Extraction

**`ColorExtractor` class:**
- **`extract_from_images(image_paths, n_colors=5)`:**
  - Resizes each image to 100x100
  - Combines all pixels, samples 30,000 if too many
  - Runs `sklearn.cluster.KMeans(n_clusters=5)`
  - Returns 5 hex color strings sorted by frequency
- **`get_accent_color(hex_colors)`:** Scores colors by `saturation * 100 + brightness_centrality * 30`, preferring mid-range brightness (30-220)
- **`lighten_color(hex_color, factor=0.92)`:** Blends toward white for background tint
- **`darken_color(hex_color, factor=0.6)`:** Scales RGB downward

### 6.4 `app/clip_extract.py` - CLIP Feature and Prompt Generation

**`CLIPContentExtractor` class:**

Uses CLIP to rank candidate text descriptions against the visual features of retrieved reference ads. This is how the system derives context from the dataset without hardcoded category mappings.

- **Candidate Pools:**
  - `FEATURE_POOL` (48 phrases): "Premium quality craftsmanship", "Advanced cushioning technology", "Cutting-edge processor power", etc.
  - `STYLE_POOL` (12 phrases): "dramatic studio lighting on dark background", "bright outdoor natural sunlight setting", etc.
  - `MOOD_POOL` (10 phrases): "energetic dynamic movement", "calm peaceful serenity", etc.
  - `SUBJECT_POOL` (12 phrases): "person wearing the product proudly", "athlete in dynamic action pose", etc.

- **`_rank_pool(image_features, pool, top_n)`:**
  1. Average the CLIP features of all reference images
  2. Encode each candidate text phrase through CLIP's text encoder
  3. Compute cosine similarity between average image features and each text
  4. Return top N candidates by similarity

- **`generate_diffusion_prompt(image_paths, brand, ...)`:**
  Builds a prompt like: `"professional commercial advertisement photography for Nike, athlete in dynamic action pose, bright outdoor natural sunlight setting, dramatic studio lighting, energetic dynamic movement mood, ultra realistic, 8k, sharp focus"`

### 6.5 `app/image_gen.py` - Image Generation

**`ImageGenerator` class:**

7-tier fallback chain for generating product images:

- **`_try_pollinations(prompt, width, height, model)`:**
  - URL: `https://image.pollinations.ai/prompt/{encoded}?width=...&height=...&model=...&nologo=true&seed=...`
  - Timeout: 120s, retries: 2
  - Models tried: "flux", "turbo", "flux-realism"

- **`_try_hf_inference(prompt, model_name, model_url)`:**
  - Sends POST to HuggingFace Inference API
  - Handles 503 (model loading) with 20s wait + retry
  - Requires `HF_TOKEN` environment variable

- **`_make_gradient(width, height, colors)`:**
  - Deterministic PIL-based gradient using extracted brand colors
  - Last resort fallback that always succeeds

**`LocalAdImageGenerator` class:** Additional local fallback for ad-specific gradient backgrounds.

### 6.6 `app/content_gen.py` - Text Generation, Translation, Enhancement

**`PollinationsTextGenerator` class:**
- URL: `https://text.pollinations.ai/`
- `generate(system_prompt, user_prompt)` -> plain text
- `generate_json(system_prompt, user_prompt)` -> parsed JSON dict
- Supports `jsonMode: True` for structured output

**`ContentGenerator` class:**
- `generate_product_content(brand, category, subcategory, features, tagline, query)` -> dict
- Calls `_generate_all_via_ai()` which sends a single prompt requesting JSON with all ad copy fields
- Fallback chain: JSON generation -> plain text headline -> brand name only

**`Translator` class:**
- Uses `deep_translator.GoogleTranslator` (free, no API key)
- `translate(text, target_lang)` -> translated string
- `translate_content(dict, target_lang)` -> dict with all values translated

**`ImageEnhancer` class:**
- `auto_enhance(image)` -> enhanced image
- Analyzes mean brightness and standard deviation
- Adjusts brightness, contrast, sharpness, and color saturation accordingly
- Used when user uploads a product image

### 6.7 `app/logo.py` - Logo Fetching

**`ProLogoFetcher` class:**

6-tier fallback for fetching brand logos:

| Tier | Source | Max Size |
|---|---|---|
| 1 | Website scraping (apple-touch-icon) | 512px |
| 2 | Google faviconV2 API | 256px |
| 3 | icon.horse | varies |
| 4 | DuckDuckGo icons | varies |
| 5 | AI-generated (Pollinations) | varies |
| 6 | Text-based (brand initials) | 200px |

- **`_remove_background(img)`:** Detects background color from border pixels, creates alpha mask, feathers edges for smooth anti-aliasing
- **`_polish_logo(logo)`:** Crops to content, resizes to 200px height
- **`BRAND_DOMAINS` dict (40+ entries):** Maps brand names to domains (Nike -> nike.com, Samsung_mobiles -> samsung.com)

### 6.8 `app/designer.py` - Ad Composition

**`ProAdDesigner` class:**

Composes the final 1080x1080 PNG ad image.

**6 Layout Themes:**

1. **`minimal_clean`** - Product hero image at top, decorative wave divider, centered headline and features below on light background
2. **`bold_hero`** - Large product image top half, dark accent-colored bottom section, bold uppercase headline, features in grid
3. **`premium_dark`** - Gradient fade from product to dark background, elegant centered typography, sophisticated feel
4. **`split_layout`** - Product image on left half, text content on right half with accent background color
5. **`card_float`** - Product image top with curved bottom edge, floating card with accent-colored text area below
6. **`gradient_mesh`** - Product image top, vibrant multi-color mesh gradient below with overlaid text

**Theme Selection Algorithm:**
```
reference_ads -> analyze(brightness, saturation, edge_complexity) -> weight_probabilities -> random_choice
```
- Dark reference ads (low brightness) -> premium_dark, bold_hero weighted higher
- Light reference ads -> minimal_clean, split_layout weighted higher
- High saturation ads -> gradient_mesh, card_float weighted higher

**Font system:**
- Caches loaded fonts by (path, size, ttc_index) key
- Detects script (Latin, Indic, CJK, Arabic) from text content
- Falls back gracefully through system fonts

**Text rendering utilities:**
- `_word_wrap()` - Wraps text to fit within max width
- `_draw_text_shadow()` - Adds shadow effect behind text
- `_draw_text_outline()` - Adds outline/stroke effect
- `_round_corners()` - Rounds image corners with alpha channel

### 6.9 `app/database.py` - SQLite Product Catalog

**`Database` class:**

**Tables:**
1. **`products`** - id (UUID), name, description, price, category, brand, image_paths (JSON), enhanced_image_path, created_at, updated_at
2. **`generated_content`** - id (UUID), product_id (FK), content_type, language, content_json, pamphlet_path, product_image_path, created_at
3. **`clicks`** - id (auto), product_id (FK), platform, source, clicked_at

**Key operations:**
- `create_product()`, `get_product()`, `list_products()`, `delete_product()` (cascades)
- `save_generated_content()` - stores AI-generated content per product
- `track_click()` / `get_analytics()` - click tracking by platform (WhatsApp, Instagram, website)

### 6.10 `app/dataset_enhancer.py` - Dataset Enhancement

(Detailed in [Section 9](#9-dataset-enhancement-self-improving-loop))

### 6.11 `app/main.py` - FastAPI Web Server

**Setup:**
- FastAPI v2.1.0 with CORS enabled (all origins)
- Static file serving for `app/static/`, `outputs/`, `uploads/`
- Lazy pipeline loading (initialized on first API request)

**Frontend:** Single-page HTML app at `app/static/index.html`

### 6.12 `app/pipeline.py` - Pipeline Orchestrator

**`AdCraftPipeline` class (singleton):**

Wires all modules together and orchestrates the 7-stage pipeline.

**Initialization:**
1. Load CLIP model (`openai/clip-vit-base-patch32`)
2. Load FAISS index (61,576 vectors)
3. Build brand index from metadata (495 brands)
4. Initialize all sub-modules (ColorExtractor, ProAdDesigner, ImageGenerator, ContentGenerator, Translator, ImageEnhancer, Database, DatasetEnhancer)

**Main method: `generate(query, languages, uploaded_image, product_id, brand_override)`**
- Executes all 7 stages sequentially
- Records timing for each stage
- Saves result JSON to `outputs/result_{timestamp}.json`
- Returns `GenerationResult` dataclass

---

## 8. Technologies and Libraries Used

### 7.1 Core ML / AI

| Library | Version | Purpose |
|---|---|---|
| `torch` | >= 2.0.0 | PyTorch deep learning framework (CLIP model inference) |
| `torchvision` | >= 0.15.0 | Image transforms for PyTorch |
| `transformers` | >= 4.30.0 | HuggingFace library (loads CLIPModel, CLIPProcessor) |
| `faiss-cpu` | >= 1.7.4 | Facebook AI Similarity Search (vector index for retrieval) |

### 7.2 Image Processing

| Library | Version | Purpose |
|---|---|---|
| `Pillow` | >= 9.0.0 | Image loading, manipulation, drawing, font rendering |
| `opencv-python` | >= 4.7.0 | Additional image processing utilities |

### 7.3 Data Processing

| Library | Version | Purpose |
|---|---|---|
| `numpy` | >= 1.24.0 | Numerical arrays for embeddings and pixel data |
| `pandas` | >= 2.0.0 | CSV reading/writing for metadata |
| `scikit-learn` | >= 1.3.0 | KMeans clustering for color extraction |

### 7.4 Web Framework

| Library | Version | Purpose |
|---|---|---|
| `fastapi` | >= 0.100.0 | REST API backend |
| `uvicorn` | >= 0.22.0 | ASGI web server |
| `python-multipart` | >= 0.0.6 | File upload support in FastAPI |

### 7.5 External APIs / Services (all free)

| Service | Usage | API Key Required |
|---|---|---|
| Pollinations.ai (text) | AI text generation for all ad copy | No |
| Pollinations.ai (image) | AI image generation (FLUX models) | No |
| HuggingFace Inference | Fallback image generation (SDXL, SD 2.1) | Optional (HF_TOKEN) |
| Google Translator | Multi-language translation (via deep_translator) | No |
| Google faviconV2 | Logo fetching | No |
| icon.horse | Logo fetching fallback | No |
| DuckDuckGo icons | Logo fetching fallback | No |

### 7.6 Utilities

| Library | Version | Purpose |
|---|---|---|
| `requests` | >= 2.28.0 | HTTP calls to Pollinations, HuggingFace, logo APIs |
| `deep-translator` | >= 1.11.0 | Google Translate wrapper (free, no key) |
| `python-dotenv` | >= 1.0.0 | Load .env file for environment variables |
| `tqdm` | >= 4.65.0 | Progress bars for data processing scripts |

### 7.7 ML Model Used

| Model | Source | Embedding Dim | Purpose |
|---|---|---|---|
| `openai/clip-vit-base-patch32` | HuggingFace | 512 | Encodes images and text into a shared vector space for similarity search |

---

## 9. API Endpoints

The FastAPI server (`app/main.py`) exposes these endpoints:

### Ad Generation

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/generate` | POST | `prompt` (required), `languages` (comma-separated), `brand`, `image` (file) | Full ad generation pipeline. Returns pamphlet URL, product image URL, text content, translations, colors, timings |

### Save & Share

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/save-generated` | POST | `name`, `description`, `price`, `category`, `brand`, `content_json`, `pamphlet_path`, `product_image_path`, `languages_generated` | Save a generated ad as a product with all content, translations, and images. Returns `product_id` and `hub_url` for immediate sharing |

### Product Catalog

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/products` | POST | `name`, `description`, `price`, `category`, `brand`, `images` (files, max 3) | Create a new product |
| `/api/products` | GET | - | List all products |
| `/api/products/{id}` | GET | - | Get product details + generated content + analytics |
| `/api/products/{id}` | DELETE | - | Delete product (cascades to content + clicks) |
| `/api/products/{id}/generate` | POST | `prompt`, `languages` | Generate ad for an existing product |

### Smart Prompt Parser

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/parse-prompt` | POST | `prompt` (required), `use_ai` (optional, "true"/"false") | Extract structured product fields from free-text. Returns extracted fields, category, completeness, missing fields. Default is fast local mode (<50ms); set `use_ai=true` for Pollinations AI refinement (~3-8s) |
| `/api/validate-fields` | POST | `parsed_data` (JSON string), `additional_fields` (JSON string) | Merge user-provided field values into previously parsed data and recalculate completeness |
| `/api/categories` | GET | - | List all 10 supported product categories with required and optional field definitions |

### Content Generation

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/describe` | POST | `prompt`, `image` (optional file) | Generate product description |
| `/api/captions` | POST | `prompt`, `languages` | Generate multi-language captions |
| `/api/enhance` | POST | `image` (file) | Auto-enhance image brightness/contrast/sharpness |

### System

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check (returns version, status) |
| `/api/languages` | GET | List all 25 supported languages |
| `/api/stats` | GET | Pipeline stats (total vectors, brands, device, products) |

### Analytics & Sharing

| Endpoint | Method | Description |
|---|---|---|
| `/hub/{product_id}` | GET | Public shareable product page with 8-platform social sharing (WhatsApp, Instagram, Facebook, X/Twitter, LinkedIn, Telegram, Pinterest, Email) |
| `/api/track/{product_id}` | POST | Track click by platform (whatsapp, instagram, facebook, twitter, linkedin, telegram, pinterest, email, website) |
| `/api/analytics/{product_id}` | GET | Get click analytics for a product by platform |
| `/file` | GET | Serve generated files (images, JSON) from allowed directories |

---

## 10. What From the Dataset Is Actually Used and Where

### Summary Table

| Stage | What is accessed | Actual image files opened? |
|---|---|---|
| Startup: Brand index building | `id_to_metadata` (all 61,576 entries scanned for brand/category) | No |
| Stage 0: Brand Matching | Brand index (in-memory dict of 495 brands) | No |
| Stage 1: FAISS Retrieval | FAISS index vectors (pre-computed 512-D embeddings) | No |
| Stage 2: Color Extraction | **Opens up to 5 retrieved ad images from disk** | **Yes** |
| Stage 3: CLIP Analysis | **Opens up to 5 retrieved ad images from disk** | **Yes** |
| Stage 3: Thumbnails | **Opens up to 3 retrieved ad images** | **Yes** |
| Stage 4: Image Generation | Uses prompt string only | No |
| Stage 5: Translation | No dataset access | No |
| Stage 6: Composition | Uses thumbnails already loaded in Stage 3 | No (already loaded) |
| Stage 7: Enhancement | Writes new image, reads no existing ones | No |

### Key Insight

**All 4 source folders are used equally.** The FAISS index was built from all 61,576 images across Advert_Gallery, Epaper1, Epaper2, and OnlineAds. At query time, any image from any source can be retrieved as a reference ad.

However, the **Advert_Gallery** images (1,951 images with clean brand labels in 167 brand subfolders) tend to dominate brand-specific queries because their brand metadata is the most accurate. Epaper and OnlineAds images contribute more to category-level or visual-similarity retrieval.

### What Happens If Image Files Are Missing

| Component | Behavior if files missing |
|---|---|
| FAISS index + metadata | Still works (pre-computed vectors exist) |
| Brand matching | Still works (built from metadata, not files) |
| Color extraction | Falls back to default palette |
| CLIP feature ranking | Falls back to first items in each pool |
| Thumbnails on final ad | No thumbnails shown |

---

## 10.5 Save & Share Flow (Ad → Product → Social Media)

After generating an ad, users can save it as a product and share it across all social media platforms in one seamless flow:

### How It Works

```
Generate Ad Creative (Step 4)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  "Save to Products & Get Share Link"                            │
│  POST /api/save-generated                                       │
│  → Creates product in catalog with all generated content         │
│  → Stores pamphlet, captions, translations, hashtags             │
│  → Returns product_id + hub_url                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Social Media Share Grid (8 platforms)                           │
│                                                                  │
│  WhatsApp  │  Instagram  │  Facebook  │  X (Twitter)            │
│  LinkedIn  │  Telegram   │  Pinterest │  Email                  │
│                                                                  │
│  Each share button:                                              │
│  1. Tracks click via POST /api/track/{product_id}               │
│  2. Opens platform-specific share dialog with:                   │
│     - Hub page URL for the product                              │
│     - Pre-filled product title and description                  │
│     - Pamphlet image URL (for Pinterest)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Hub Page (/hub/{product_id})                                    │
│  Public shareable page with:                                     │
│  - Hero image (pamphlet or product image)                       │
│  - Brand badge, title, price, description                       │
│  - Copyable Instagram caption and WhatsApp message              │
│  - 8-platform social share buttons                              │
│  - Engagement analytics (clicks by platform)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Social Media Platforms

| Platform | Share Method | What's Shared |
|----------|-------------|---------------|
| WhatsApp | `api.whatsapp.com/send` with pre-filled text | Product title + hub URL |
| Instagram | Copy link to clipboard + user instruction | Hub URL (user pastes in post/story) |
| Facebook | `facebook.com/sharer/sharer.php` popup | Hub URL with OG meta preview |
| X (Twitter) | `twitter.com/intent/tweet` popup | Product title + hub URL |
| LinkedIn | `linkedin.com/sharing/share-offsite` popup | Hub URL with preview |
| Telegram | `t.me/share/url` with text | Product title + hub URL |
| Pinterest | `pinterest.com/pin/create/button` popup | Hub URL + pamphlet image + description |
| Email | `mailto:` link | Subject: product title, Body: description + hub URL |

### Analytics Tracking

Every social share button click is tracked via `POST /api/track/{product_id}?platform=<name>`. Analytics are viewable:
- In the product detail page (Products → View → Traffic Analytics)
- On the hub page itself (Total Engagement count)
- Via API: `GET /api/analytics/{product_id}`

---

## 11. Dataset Enhancement (Self-Improving Loop)

After every successful ad generation, the system feeds the generated pamphlet back into the dataset.

### How It Works

**Step 1 - Save to dataset folder:**
- Looks up where existing images for the brand live on disk (via `id_to_metadata`)
- Copies the generated pamphlet into that same folder
- If the brand is new, creates a folder under `data/images/Advert_Gallery/NewsPaperAds/Advert_Gallery/{brand}/`
- Filename format: `{Brand}_gen_{timestamp}.png`

**Step 2 - Generate CLIP embedding:**
- Opens the new pamphlet image
- Processes through CLIP vision encoder
- L2-normalizes to get a 512-D vector

**Step 3 - Update FAISS index (thread-safe):**
- Adds the new vector to the in-memory FAISS index
- Adds metadata entry to `id_to_metadata` dict
- Updates the brand matcher's brand index (adds new index ID to the brand's list)

**Step 4 - Update metadata CSV:**
- Appends a new row to `processed/metadata/madverse_metadata.csv`

**Step 5 - Persist to disk:**
- Writes updated FAISS index to `madverse_index.faiss`
- Writes updated metadata to `id_to_metadata.pkl`

### Categorization Approach

The new image inherits its **category and subcategory from the matched brand**, not from independent classification of the item. For example:
- Query "Nike perfume" -> brand=Nike -> category inherited from Nike's historical data (likely "sports") -> the perfume ad gets labeled as "sports"

### Thread Safety

All FAISS index updates and CSV writes are protected by `threading.Lock()`. The entire `enhance()` method is wrapped in try/except, so failures never crash the main pipeline.

### Pros of Dataset Enhancement

- **Self-improving:** The dataset grows with every generation, improving future retrieval
- **Consistent taxonomy:** New images follow the same brand/category structure as existing data
- **Incremental:** No need to rebuild the full FAISS index
- **Persistent:** Survives server restarts (written to disk)
- **Non-blocking:** Failures are logged but never disrupt the user's generation

### Cons of Dataset Enhancement

- **No independent item classification:** Category is tied to brand, not the actual product content
- **No quality gate:** Bad generated images still get added to the index
- **Category drift risk:** A brand's category is fixed by historical majority; new categories for the same brand are ignored
- **Single-label taxonomy:** Each brand gets exactly one category, even if the brand spans multiple (e.g., Samsung: phones, TVs, appliances)
- **No deduplication:** Running the same query twice adds two near-identical entries
- **Pickle-based persistence:** `id_to_metadata.pkl` is fragile with no backup or versioning
- **CSV column alignment:** Appends rows with its own column order, which could mismatch if the CSV structure changes

---

## 12. File Structure

```
MAdVerse/
├── app/                              # Main application package
│   ├── __init__.py                   # Package marker
│   ├── main.py                       # FastAPI backend (API endpoints, file serving, CORS)
│   ├── pipeline.py                   # Pipeline orchestrator (7 stages, singleton)
│   ├── models.py                     # Data classes, constants, path config, language list
│   ├── brands.py                     # Brand taglines dict + BrandMatcher (fuzzy matching)
│   ├── colors.py                     # ColorExtractor (KMeans clustering on reference ads)
│   ├── clip_extract.py               # CLIPContentExtractor (CLIP-based feature/prompt ranking)
│   ├── logo.py                       # ProLogoFetcher (6-tier logo fallback chain)
│   ├── image_gen.py                  # ImageGenerator (7-tier image generation fallback)
│   ├── content_gen.py                # ContentGenerator, Translator, ImageEnhancer
│   ├── smart_prompt.py               # SmartPromptParser (field extraction, category detection)
│   ├── designer.py                   # ProAdDesigner (6 themes, multi-script fonts)
│   ├── database.py                   # SQLite product catalog + click analytics
│   ├── dataset_enhancer.py           # DatasetEnhancer (self-improving loop)
│   └── static/
│       └── index.html                # Web frontend (single-page app)
│
├── data/
│   ├── annotations/                  # JSON annotation files (~13 MB)
│   │   ├── adgal_annot_j.json
│   │   ├── web_annot_j.json
│   │   ├── epaper1_annotation.json
│   │   └── epaper2_annotation.json
│   └── images/                       # ~61,626 ad images (~35 GB)
│       ├── Advert_Gallery/           # 1,951 images in 167 brand folders
│       ├── OnlineAds/                # 22,397 images in 11 category folders
│       ├── Epaper1/                  # 20,564 images in 8 language folders
│       └── Epaper2/                  # 16,683 images in 10 language folders
│
├── embeddings/
│   ├── image_embeddings.pkl          # All CLIP embeddings (~281 MB)
│   └── faiss_indexes/
│       ├── madverse_index.faiss      # FAISS vector index (61,576 vectors x 512 dims)
│       ├── id_to_metadata.pkl        # Index ID -> image metadata mapping
│       └── index_stats.json          # Index statistics
│
├── processed/
│   └── metadata/
│       └── madverse_metadata.csv     # 61,595 rows of structured metadata
│
├── outputs/                          # Generated ads (auto-created at runtime)
├── uploads/                          # User uploads (auto-created at runtime)
├── products_db/                      # SQLite database (auto-created at runtime)
│
├── DatasetLoad.py                    # Step 1: Annotations + images -> metadata CSV
├── GenerateEmbeddings.py             # Step 2: Images -> CLIP 512-D embeddings
├── BuildFAISS.py                     # Step 3: Embeddings -> FAISS index
├── RAGPipeline.py                    # Legacy standalone pipeline (reference)
├── run.py                            # Web server launcher (uvicorn)
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (HF_TOKEN, etc.)
└── README.md                         # Project documentation
```

---

## 13. How to Run

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for embedding generation, not required for runtime)
- ~35 GB disk space for dataset images
- No API keys required for basic operation

### One-Time Setup (Index Building)

Only needed if building from scratch. Pre-built indexes are included.

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Build metadata CSV from annotations + images
python DatasetLoad.py

# Step 2: Generate CLIP embeddings for all images (needs HF_TOKEN)
export HF_TOKEN=your_huggingface_token
python GenerateEmbeddings.py

# Step 3: Build FAISS index from embeddings
python BuildFAISS.py
```

### Running the Web Application

```bash
python run.py
```

Open **http://localhost:8000** in your browser.
API docs available at **http://localhost:8000/docs**.

### Optional Environment Variables

```bash
HF_TOKEN=<huggingface_token>       # For HuggingFace model download and inference fallback
TOGETHER_API_KEY=<key>             # Optional, not actively used
GOOGLE_API_KEY=<key>               # Optional, for enhanced logo fetching
```

---

## 14. Pros and Cons

### Pros

| Area | Detail |
|---|---|
| **No hardcoded templates** | All ad text is AI-generated per request. All visual decisions (colors, style, mood, theme) are derived from the actual retrieved reference ads via CLIP. No category-to-template lookup tables. |
| **Free-tier friendly** | Core pipeline requires zero paid API keys. Pollinations.ai, deep_translator, and logo APIs are all free. |
| **Self-improving dataset** | Every generated ad is fed back into the FAISS index, making the dataset richer over time. |
| **Extensive fallback chains** | Image generation: 7 tiers. Logo fetching: 6 tiers. Text generation: 3 tiers. The system degrades gracefully rather than failing. |
| **Multi-language support** | 25+ languages with native script rendering. Bilingual ad output when two languages are requested. |
| **Modular architecture** | Each stage is a separate module (brands.py, colors.py, clip_extract.py, etc.) that can be tested independently. |
| **Thread-safe dataset updates** | FAISS index and metadata writes are protected by threading locks. |
| **Product catalog** | Built-in SQLite database for managing products, generated content, and click analytics. |
| **Shareable hub pages** | Each product gets a public URL with analytics tracking. |

### Cons

| Area | Detail |
|---|---|
| **Brand-first categorization** | Category is derived from the brand, not the actual item. "Nike perfume" would get categorized as "sports" because that's Nike's historical majority category. |
| **Fuzzy matching false positives** | The 0.5 cutoff on `difflib.SequenceMatcher` is low. "Pixar" could match "Puma", "Bata" could match "Data". |
| **No quality gate for generated content** | Every generated pamphlet is added to the dataset regardless of quality. Bad outputs pollute the FAISS index over time. |
| **Single-label brand taxonomy** | Each brand maps to exactly one category and subcategory. Multi-category brands (Samsung: phones + TVs + appliances) are flattened to their most common one. |
| **No deduplication** | Running the same query multiple times adds near-identical entries to the FAISS index. |
| **Pickle-based persistence** | `id_to_metadata.pkl` is fragile. Corruption means losing all metadata. No backup strategy. |
| **Sequential pipeline** | All 7 stages run sequentially. Stages 2 and 3 (which both open the same images) could potentially run in parallel. |
| **External API dependency for text** | All ad copy depends on Pollinations.ai being available. If it's down, text generation falls back to just the brand name with no features, tagline, or CTA. |
| **CSV append fragility** | `_append_to_csv` writes rows with its own column order. If the metadata CSV schema changes, new rows become misaligned. |
| **IndexFlatL2 (exact search)** | With 61,576 vectors this is fine, but won't scale well to millions of images. Would need to switch to IVF or HNSW index for larger datasets. |
| **No image deduplication in dataset** | The original dataset may contain duplicate or near-duplicate images, and generated images add more redundancy. |

---

*This documentation covers the complete MAdVerse pipeline as of the `develop` branch on 25 February 2026.*
