#!/bin/bash
# Docker startup script with FAISS validation
# This runs before the main application starts in Docker

set -e  # Exit on error

echo "=========================================="
echo "  MAdVerse Startup - Validating Setup"
echo "=========================================="
echo ""

# Function to check if file is a Git LFS pointer
is_lfs_pointer() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1  # File doesn't exist
    fi
    
    # Check file size (LFS pointers are < 200 bytes)
    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    
    if [ "$size" -lt 200 ]; then
        # Check if it starts with LFS header
        if head -n 1 "$file" 2>/dev/null | grep -q "version https://git-lfs.github.com"; then
            return 0  # Is an LFS pointer
        fi
    fi
    
    return 1  # Not an LFS pointer
}

# Check critical files
echo "[CHECK] Validating FAISS index files..."

FAISS_INDEX="embeddings/faiss_indexes/madverse_index.faiss"
FAISS_METADATA="embeddings/faiss_indexes/id_to_metadata.pkl"

if [ ! -f "$FAISS_INDEX" ]; then
    echo "❌ ERROR: FAISS index not found at $FAISS_INDEX"
    echo ""
    echo "SOLUTION: Run 'git lfs pull' before building Docker image"
    exit 1
fi

if is_lfs_pointer "$FAISS_INDEX"; then
    echo "❌ ERROR: FAISS index is a Git LFS pointer, not the actual file!"
    echo ""
    echo "This means Git LFS files were not downloaded."
    echo ""
    echo "SOLUTION:"
    echo "  1. Install Git LFS: https://git-lfs.github.com/"
    echo "  2. Run: git lfs install"
    echo "  3. Run: git lfs pull"
    echo "  4. Rebuild Docker: docker-compose up -d --build"
    echo ""
    exit 1
fi

# Check file size
FAISS_SIZE=$(stat -f%z "$FAISS_INDEX" 2>/dev/null || stat -c%s "$FAISS_INDEX" 2>/dev/null || echo "0")
MIN_SIZE=$((100 * 1024 * 1024))  # 100 MB minimum

if [ "$FAISS_SIZE" -lt "$MIN_SIZE" ]; then
    echo "❌ ERROR: FAISS index file is too small ($FAISS_SIZE bytes)"
    echo "   Expected at least 100 MB. File may be corrupted."
    echo ""
    echo "SOLUTION: Re-pull from Git LFS:"
    echo "  git lfs pull"
    echo "  docker-compose up -d --build"
    exit 1
fi

echo "✓ FAISS index OK ($(($FAISS_SIZE / 1024 / 1024)) MB)"

# Check metadata
if [ ! -f "$FAISS_METADATA" ]; then
    echo "❌ ERROR: Metadata file not found at $FAISS_METADATA"
    exit 1
fi

if is_lfs_pointer "$FAISS_METADATA"; then
    echo "❌ ERROR: Metadata is a Git LFS pointer!"
    echo "   Run 'git lfs pull' before building Docker image"
    exit 1
fi

echo "✓ Metadata file OK"

# Create runtime directories
echo ""
echo "[SETUP] Creating runtime directories..."
mkdir -p outputs uploads products_db
echo "✓ Directories created"

# Check environment variables
echo ""
echo "[CONFIG] Checking environment variables..."

if [ -z "$HF_TOKEN" ]; then
    echo "⚠  WARNING: HF_TOKEN not set"
    echo "   Image generation will use fallback (lower quality)"
else
    echo "✓ HF_TOKEN configured"
fi

if [ -z "$GOOGLE_API_KEY" ] && [ -z "$GROQ_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠  WARNING: No text generation API keys found"
    echo "   Will use free Pollinations.ai (works but slower)"
else
    echo "✓ Text generation API key(s) configured"
fi

echo ""
echo "=========================================="
echo "  ✓ Validation complete! Starting app..."
echo "=========================================="
echo ""

# Start the application
exec "$@"
