#!/bin/bash
# ============================================================
# MAdVerse (AdCraft AI) — Startup Script for HF Spaces / Docker
# Downloads dataset from Zenodo on first boot, then starts server.
# ============================================================

set -e

ZENODO_BASE="https://zenodo.org/records/10657763/files"
DATA_DIR="/app/data/images"
ANNOTATIONS_DIR="/app/data/annotations"
MARKER_FILE="/app/data/images/.download_complete"
DB_FILE="${MADVERSE_DB_PATH:-/app/products_db/adcraft.db}"
HF_CACHE_DIR="${HF_HOME:-/app/.cache/huggingface}"

echo "============================================================"
echo "  MAdVerse AI — Container Startup"
echo "============================================================"

# Load environment variables from /app/.env when present (useful with bind mounts)
if [ -f "/app/.env" ]; then
    echo "[STARTUP] Loading environment from /app/.env"
    set -a
    # shellcheck disable=SC1091
    . /app/.env
    set +a
else
    echo "[STARTUP] /app/.env not found (this is normal if using --env-file)."
fi

# Create required directories
mkdir -p /app/outputs /app/uploads /app/products_db /app/data/images /app/data/indices
mkdir -p "${HF_CACHE_DIR}"

echo "[STARTUP] HuggingFace cache directory: ${HF_CACHE_DIR}"
if [ -d "${HF_CACHE_DIR}/hub" ] && [ "$(ls -A "${HF_CACHE_DIR}/hub" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[STARTUP] HuggingFace cache detected. Model downloads should be skipped when available."
else
    echo "[STARTUP] HuggingFace cache is empty. First run may download CLIP model weights."
fi

if [ -n "${HF_TOKEN:-}" ] || [ -n "${TOGETHER_API_KEY:-}" ]; then
    echo "[STARTUP] Image generation API key status: configured (HF and/or Together)."
else
    echo "[STARTUP] Image generation API key status: not configured -> local gradient fallback may produce solid color images."
fi

# Optional one-shot reset for stale Docker volume DB data
if [ "${MADVERSE_RESET_DB_ON_START:-0}" = "1" ]; then
    echo "[STARTUP] MADVERSE_RESET_DB_ON_START=1 -> removing DB file: ${DB_FILE}"
    rm -f "${DB_FILE}"
fi

# ------------------------------------------------------------------
# Step 1: Download dataset images from Zenodo (only on first boot)
# ------------------------------------------------------------------
# Skip download if SKIP_DATASET_DOWNLOAD=1 (embeddings work without images)
if [ "${SKIP_DATASET_DOWNLOAD:-0}" = "1" ]; then
    echo "[STARTUP] Dataset download SKIPPED (SKIP_DATASET_DOWNLOAD=1)"
    echo "[STARTUP] Embeddings and FAISS index will be used for generation."
    echo "[STARTUP] Original dataset images not needed for ad generation."
    touch "$MARKER_FILE"
elif [ -f "$MARKER_FILE" ]; then
    echo "[STARTUP] Dataset already downloaded. Skipping."
else
    echo "[STARTUP] Downloading MAdVerse dataset from Zenodo..."
    echo "[STARTUP] This only happens on the FIRST boot (~10-15 min)."
    echo ""

    cd /tmp

    # Download and extract each image archive
    for archive in Advert_Gallery Epaper1 Epaper2 OnlineAds; do
        echo "[DOWNLOAD] Fetching ${archive}.zip ..."
        curl -L --retry 3 --retry-delay 5 -o "${archive}.zip" \
            "${ZENODO_BASE}/${archive}.zip?download=1" 2>&1 | tail -1
        
        echo "[EXTRACT]  Unzipping ${archive}.zip ..."
        unzip -q -o "${archive}.zip" -d "$DATA_DIR/" 2>/dev/null || true
        rm -f "${archive}.zip"
        echo "[DONE]     ${archive} extracted."
        echo ""
    done

    # Download annotation files (if not already baked into image)
    if [ ! -f "$ANNOTATIONS_DIR/web_annot_j.json" ]; then
        echo "[DOWNLOAD] Fetching annotation files..."
        for annot in adgal_annot_j.json epaper1_annotation.json epaper2_annotation.json web_annot_j.json dataset_readme.md; do
            curl -L --retry 3 -o "$ANNOTATIONS_DIR/$annot" \
                "${ZENODO_BASE}/${annot}?download=1" 2>&1 | tail -1
        done
    fi

    # Mark download as complete
    touch "$MARKER_FILE"
    echo ""
    echo "[STARTUP] Dataset download complete!"
fi

# ------------------------------------------------------------------
# Step 2: Start the FastAPI server
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Starting MAdVerse AI Server"
echo "  Port: ${PORT:-7860}"
echo "============================================================"
echo ""

cd /app

exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-7860}" \
    --log-level info \
    --no-access-log
