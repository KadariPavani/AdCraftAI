---
title: MAdVerse - AI Ad Generation Platform
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - advertising
  - ai
  - generative-ai
  - text-to-image
  - fastapi
short_description: Generate professional ad creatives with AI-powered design
---

# MAdVerse (AdCraft AI) 🎨

**AI-powered ad creative generation platform** that transforms product descriptions into polished, multi-platform ad campaigns.

## 🚀 What It Does

- **🎯 Smart Prompt Parsing**: Converts free-form product descriptions into structured ad metadata
- **🖼️ Image Generation**: Creates professional product visuals with AI (FLUX.1-schnell)
- **✍️ Multi-Platform Copy**: Generates Instagram captions, WhatsApp messages, and hashtags
- **🌍 Multi-Language**: Supports 50+ languages with automatic translation
- **📊 Dataset Intelligence**: Leverages 50,000+ curated ads with semantic search (FAISS)
- **💾 Product Hub**: Save and share generated ads with engagement tracking

## 🎥 Demo

Try it yourself! Enter a product description like:
> "Wireless headphones with noise cancellation, perfect for commuters, priced at $149"

The app will:
1. Parse the description and extract key fields
2. Find relevant design inspiration from 50K+ ads
3. Generate product image
4. Create ad copy for Instagram and WhatsApp
5. Produce multi-language captions
6. Save as a shareable product page

## 📚 Tech Stack

- **Framework**: FastAPI + Python 3.11
- **ML/AI**: 
  - PyTorch + Transformers (CLIP embeddings)
  - FAISS (semantic search)
  - FLUX.1-schnell (image generation)
  - Google Gemini / Groq (text generation)
- **Image Processing**: Pillow, OpenCV
- **Translation**: Deep Translator (50+ languages)
- **Database**: SQLite (product storage & analytics)

## 🔑 Setup (Required)

For best results, add these secrets in **Settings → Repository secrets**:

### Required (at least one for text generation):
- `GOOGLE_API_KEY` - Google Gemini API key ([Get free key](https://aistudio.google.com/apikey))
- `GROQ_API_KEY` - Groq API key ([Get free key](https://console.groq.com))

### Optional (improved image quality):
- `HF_TOKEN` - Hugging Face token ([Get free token](https://huggingface.co/settings/tokens))
- `TOGETHER_API_KEY` - Together AI key

**Without API keys**: The app works with free alternatives (Pollinations.ai) but quality may be lower.

## 📖 How to Use

1. **Home Page** (`/`): Main ad generation interface
   - Enter product description
   - Upload optional product image
   - Select target language
   - Generate complete ad campaign

2. **Dataset Explorer** (`/dataset`): Browse the 50K+ ad dataset
   - View statistics by category
   - Explore supported languages
   - See dataset composition

3. **Product Hub** (`/hub/{product_id}`): Share generated ads
   - Public shareable links
   - Track engagement by platform
   - View all generated content

## 🏗️ Architecture

```
app/
├── main.py              # FastAPI routes & UI
├── pipeline.py          # Ad generation orchestrator
├── smart_prompt.py      # Prompt parsing & field extraction
├── image_gen.py         # Image generation (FLUX + fallbacks)
├── content_gen.py       # Text generation & translation
├── designer.py          # Layout composition
└── database.py          # SQLite product storage

embeddings/              # Pre-computed CLIP embeddings (410 MB)
├── image_embeddings.pkl
└── faiss_indexes/
    ├── madverse_index.faiss
    └── id_to_metadata.pkl

data/annotations/        # Dataset metadata (12 MB)
```

## 🎯 Key Features

### 1. **Semantic Search** (FAISS)
- 50,000+ ads indexed with CLIP embeddings
- Find visually similar designs for any product
- Category-aware recommendations

### 2. **Multi-Model Fallback System**
- **Text**: Gemini → Groq → Claude → Pollinations.ai
- **Images**: FLUX.1-schnell → Together AI → Local gradients
- Ensures high availability

### 3. **Smart Prompt Parser**
Extract structured data from natural language:
```
Input: "Premium leather wallet, handcrafted, $79.99, perfect gift"
Output: {
  "title": "Premium Leather Wallet",
  "category": "Fashion Accessories",
  "price": "$79.99",
  "occasion": "Gift",
  "material": "Leather"
}
```

### 4. **Multi-Language Support**
- Automatic translation to 50+ languages
- Language-specific caption styles
- Preserves brand voice across markets

## 📊 Dataset

Based on **MAdVerse** ([Zenodo](https://zenodo.org/records/10657763)) - a curated collection of 50,000+ ads from:
- Web platforms (e-commerce, social media)
- Print media (newspapers, magazines)
- Multiple categories (fashion, electronics, food, etc.)

**Note**: This Space uses **embeddings only** (no images downloaded). Full dataset available at source.

## 🛠️ Local Development

```bash
# Clone and setup
git lfs install
git clone https://huggingface.co/spaces/YOUR_USERNAME/madverse
cd madverse
git lfs pull

# Run with Docker
docker-compose up -d --build

# Or run locally
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Access at `http://localhost:8000`

## 📝 API Endpoints

- `GET /` - Main UI
- `GET /dataset` - Dataset explorer
- `GET /hub/{id}` - Product hub page
- `POST /api/generate` - Generate ad creative
- `POST /api/parse-prompt` - Parse product description
- `POST /api/enhance` - Enhance product image
- `POST /api/describe` - Generate product description
- `POST /api/captions` - Generate multi-language captions
- `GET /api/stats` - Dataset statistics

Full API docs: `http://localhost:8000/docs`

## 🐛 Troubleshooting

**Issue**: "FAISS index not found"
- **Cause**: Git LFS files not downloaded
- **Fix**: Run `git lfs install && git lfs pull` before deploying

**Issue**: Low quality images
- **Cause**: No HF_TOKEN configured
- **Fix**: Add HF_TOKEN in Space settings

**Issue**: Slow text generation
- **Cause**: Using free fallback (Pollinations.ai)
- **Fix**: Add GOOGLE_API_KEY or GROQ_API_KEY

## 📜 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Credits

- **Dataset**: MAdVerse by Tripathi et al. ([Paper](https://arxiv.org/abs/2311.09534))
- **Models**: CLIP (OpenAI), FLUX.1-schnell (Black Forest Labs), Gemini (Google)
- **Framework**: FastAPI, PyTorch, Hugging Face Transformers

## 🔗 Links

- **GitHub**: [Original Repository](#)
- **Paper**: [MAdVerse: A Hierarchically Labeled Dataset](https://arxiv.org/abs/2311.09534)
- **Dataset**: [Zenodo](https://zenodo.org/records/10657763)

---

Built with ❤️ for the AI community | Star ⭐ if you find it useful!
