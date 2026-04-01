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

echo "============================================================"
echo "  MAdVerse AI — Container Startup"
echo "============================================================"

# Create required directories
mkdir -p /app/outputs /app/uploads /app/products_db /app/data/images /app/data/indices

# ------------------------------------------------------------------
# Step 1: Download dataset images from Zenodo (only on first boot)
# ------------------------------------------------------------------
if [ -f "$MARKER_FILE" ]; then
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
