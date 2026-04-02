# ============================================================
# MAdVerse (AdCraft AI) — Dockerfile
# Optimized for Hugging Face Spaces (free tier)
# ============================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Image processing (OpenCV, Pillow)
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    # Fonts for ad designer (cross-platform replacement for Windows fonts)
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fonts-noto-core \
    # Download tools for Zenodo dataset
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────────────────────
# Install PyTorch CPU-only first (smaller image, no CUDA needed on HF Spaces)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────
COPY app/ ./app/
COPY run.py .
COPY DatasetLoad.py .
COPY GenerateEmbeddings.py .
COPY BuildFAISS.py .
COPY RAGPipeline.py .

# ── Copy data files (annotations + embeddings, NOT images) ──
# Annotations (~12 MB) — metadata for the dataset
COPY data/annotations/ ./data/annotations/

# Embeddings + FAISS index (~410 MB) — pre-computed, required at startup
COPY embeddings/ ./embeddings/

# Processed metadata (~12 MB)
COPY processed/ ./processed/

# Data indices
COPY data/indices/ ./data/indices/

# ── Copy startup script ─────────────────────────────────────
COPY scripts/start.sh /start.sh
# Fix Windows line endings (CRLF -> LF) and remove BOM
RUN sed -i 's/\r$//' /start.sh && \
    sed -i '1s/^\xEF\xBB\xBF//' /start.sh && \
    chmod +x /start.sh

# ── Create runtime directories ──────────────────────────────
RUN mkdir -p /app/outputs /app/uploads /app/products_db /app/data/images /app/.cache/huggingface

# ── Expose port ─────────────────────────────────────────────
# HF Spaces uses port 7860 by default, local dev uses 8000
EXPOSE 7860 8000

# ── Health check ─────────────────────────────────────────────
# Checks PORT env var, defaults to 7860 for HF Spaces
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/api/health || exit 1

# ── Start ────────────────────────────────────────────────────
CMD ["/start.sh"]
