#!/bin/bash
# ============================================================
# Quick Deployment Script for Hugging Face Spaces
# ============================================================

set -e

echo "================================================"
echo "  MAdVerse - Hugging Face Spaces Deployment"
echo "================================================"
echo ""

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ ERROR: Git LFS is not installed!"
    echo ""
    echo "Please install Git LFS first:"
    echo "  Windows: Download from https://git-lfs.github.com/"
    echo "  Mac:     brew install git-lfs"
    echo "  Linux:   sudo apt-get install git-lfs"
    echo ""
    exit 1
fi

echo "✓ Git LFS is installed"

# Check if .gitattributes exists
if [ ! -f ".gitattributes" ]; then
    echo "❌ ERROR: .gitattributes file not found!"
    echo "   This file is required for Git LFS tracking."
    exit 1
fi

echo "✓ .gitattributes found"

# Verify FAISS index
echo ""
echo "[VALIDATION] Checking FAISS index..."
if [ ! -f "embeddings/faiss_indexes/madverse_index.faiss" ]; then
    echo "❌ ERROR: FAISS index not found!"
    echo "   Run 'git lfs pull' to download LFS files."
    exit 1
fi

# Check if it's a Git LFS pointer
FAISS_SIZE=$(stat -f%z "embeddings/faiss_indexes/madverse_index.faiss" 2>/dev/null || stat -c%s "embeddings/faiss_indexes/madverse_index.faiss" 2>/dev/null || echo "0")
if [ "$FAISS_SIZE" -lt 1000000 ]; then
    echo "❌ ERROR: FAISS index appears to be a Git LFS pointer!"
    echo "   Size: $FAISS_SIZE bytes (expected > 100 MB)"
    echo ""
    echo "   Fix: Run 'git lfs pull' to download actual files."
    exit 1
fi

echo "✓ FAISS index OK ($(($FAISS_SIZE / 1024 / 1024)) MB)"

# Check if HF Space URL is provided
if [ -z "$1" ]; then
    echo ""
    echo "❌ ERROR: Hugging Face Space URL required!"
    echo ""
    echo "Usage:"
    echo "  ./scripts/deploy-to-hf.sh https://huggingface.co/spaces/YOUR_USERNAME/madverse"
    echo ""
    echo "Steps:"
    echo "  1. Create a Space at https://huggingface.co/new-space"
    echo "  2. Choose SDK: Docker, Hardware: cpu-basic"
    echo "  3. Copy the Space URL"
    echo "  4. Run this script with the URL"
    exit 1
fi

HF_SPACE_URL="$1"
TEMP_DIR="hf_deploy_temp"

echo ""
echo "================================================"
echo "  Deployment Target: $HF_SPACE_URL"
echo "================================================"
echo ""

# Ask for confirmation
read -p "Deploy to this Space? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Clone HF Space
echo ""
echo "[STEP 1/5] Cloning Hugging Face Space..."
rm -rf "$TEMP_DIR"
git clone "$HF_SPACE_URL" "$TEMP_DIR"
cd "$TEMP_DIR"

# Copy files (excluding unwanted directories)
echo ""
echo "[STEP 2/5] Copying project files..."
echo "  Excluding: .git, .venv, data/images, outputs, uploads, products_db"

rsync -av --exclude='.git' --exclude='.venv' --exclude='data/images' \
    --exclude='outputs' --exclude='uploads' --exclude='products_db' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    --exclude='hf_deploy_temp' \
    ../ ./ 2>/dev/null || {
    # Fallback for systems without rsync
    echo "  Using manual copy (rsync not available)..."
    cp -r ../app ./
    cp -r ../scripts ./
    cp -r ../embeddings ./
    cp -r ../processed ./
    cp -r ../data ./
    cp ../*.py ./
    cp ../*.md ./
    cp ../*.txt ./
    cp ../*.yml ./
    cp ../*.yaml ./
    cp ../Dockerfile ./
    cp ../.gitattributes ./
    cp ../.gitignore ./
    cp ../.dockerignore ./
    cp ../.env.example ./
}

# Use HF Space README
echo ""
echo "[STEP 3/5] Setting up README for Hugging Face..."
if [ -f "README_HF_SPACE.md" ]; then
    cp README.md README_GITHUB.md 2>/dev/null || true
    cp README_HF_SPACE.md README.md
    echo "  ✓ README.md updated for Hugging Face Spaces"
else
    echo "  ⚠  README_HF_SPACE.md not found, using existing README.md"
fi

# Setup Git LFS
echo ""
echo "[STEP 4/5] Configuring Git LFS..."
git lfs install

# Track large files
git lfs track "embeddings/image_embeddings.pkl"
git lfs track "embeddings/faiss_indexes/*.faiss"
git lfs track "embeddings/faiss_indexes/*.pkl"
git lfs track "processed/metadata/*.csv"
git lfs track "data/annotations/*.json"

echo "  ✓ Git LFS configured"

# Commit and push
echo ""
echo "[STEP 5/5] Committing and pushing to Hugging Face..."
git add .
git add .gitattributes  # Ensure LFS config is tracked

git commit -m "Deploy MAdVerse AI to Hugging Face Spaces

Features:
- FastAPI ad generation platform
- FAISS semantic search (50K+ ads)
- Multi-language support (50+ languages)
- Docker-based deployment
- Optimized for cpu-basic tier (FREE)

Components:
- Application code + dependencies
- Pre-computed FAISS index (345 MB)
- CLIP embeddings (65 MB)
- Dataset metadata (12 MB)

Total size: ~450 MB (within HF Spaces limits)
" || {
    echo "  No changes to commit (already up to date)"
}

echo ""
echo "  Pushing to Hugging Face (this may take 5-10 minutes for LFS files)..."
git push origin main

echo ""
echo "================================================"
echo "  ✅ Deployment Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Add API keys in Space Settings:"
echo "   $HF_SPACE_URL/settings"
echo ""
echo "   Required (at least one):"
echo "   - GOOGLE_API_KEY  (Google Gemini)"
echo "   - GROQ_API_KEY    (Groq)"
echo ""
echo "   Optional (better quality):"
echo "   - HF_TOKEN        (Hugging Face)"
echo "   - TOGETHER_API_KEY (Together AI)"
echo ""
echo "2. Monitor build progress:"
echo "   $HF_SPACE_URL/logs"
echo ""
echo "3. Build duration: ~10-15 minutes (first time)"
echo ""
echo "4. Once running, your Space will be available at:"
echo "   ${HF_SPACE_URL/spaces\//}"
echo ""
echo "================================================"
echo ""

# Cleanup
cd ..
read -p "Delete temporary deployment folder? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEMP_DIR"
    echo "✓ Cleanup complete"
fi

echo ""
echo "🎉 Happy deploying!"
