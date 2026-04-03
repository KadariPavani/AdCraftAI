# 🔧 MAdVerse Troubleshooting Guide

## Table of Contents
- [FAISS Index Errors](#faiss-index-errors)
- [Docker Issues](#docker-issues)
- [API Key Problems](#api-key-problems)
- [Generation Failures](#generation-failures)
- [Performance Issues](#performance-issues)

---

## FAISS Index Errors

### ❌ Error: `Pipeline error: Error in virtual void faiss::IndexFlat::reconstruct... 'key < ntotal' failed`

**Cause:** Git LFS files were not downloaded. You have Git LFS pointer files instead of actual data.

**How to identify:**
```bash
# Check if files are pointers
python scripts/check_faiss.py
```

**Solution:**
```bash
# 1. Install Git LFS (if not already installed)
git lfs install

# 2. Pull the actual files
git lfs pull

# 3. Verify files are correct (should show "ALL CHECKS PASSED")
python scripts/check_faiss.py

# 4. Rebuild Docker (IMPORTANT - removes old cached layers)
docker-compose down
docker-compose up -d --build

# 5. Check logs
docker-compose logs -f
```

**Prevention:** Always run `git lfs install` before cloning, or run `git lfs pull` immediately after cloning.

---

### ❌ Error: `FAISS index not found at embeddings/faiss_indexes/madverse_index.faiss`

**Cause:** Missing FAISS index file.

**Solution:**
```bash
# Check if directory exists
ls -la embeddings/faiss_indexes/

# If empty or missing, pull from Git LFS
git lfs pull

# Verify
python scripts/check_faiss.py
```

---

### ❌ Error: `Failed to load FAISS index` or corrupted index

**Cause:** Incomplete or corrupted download.

**Solution:**
```bash
# Force re-download from Git LFS
git lfs fetch --all
git lfs pull

# Verify integrity
python scripts/check_faiss.py

# Rebuild Docker
docker-compose down
docker-compose up -d --build
```

---

## Docker Issues

### ❌ Container keeps restarting

**Check logs first:**
```bash
docker-compose logs -f
```

**Common causes:**

1. **Git LFS files missing:**
   ```bash
   # Look for: "FAISS index is a Git LFS pointer"
   git lfs pull
   docker-compose up -d --build
   ```

2. **Port already in use:**
   ```bash
   # Windows: Check what's using port 8000
   netstat -ano | findstr :8000
   
   # Linux/Mac:
   lsof -i :8000
   
   # Solution: Change port in docker-compose.yml
   ports:
     - "8001:8000"  # Use 8001 instead
   ```

3. **Out of memory:**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop → Settings → Resources → Memory → 4GB+
   ```

---

### ❌ Error: `exec format error`

**Cause:** Windows line endings (CRLF) in shell scripts.

**Solution:**
Already handled in Dockerfile, but if you modified scripts:
```bash
# Convert line endings
dos2unix scripts/*.sh

# Or use sed
sed -i 's/\r$//' scripts/start.sh

# Rebuild
docker-compose up -d --build
```

---

### ❌ Slow Docker build (takes > 20 minutes)

**Normal on first build:** Downloads PyTorch, CLIP model, etc.

**Subsequent builds should be fast due to caching.**

**To speed up:**
```bash
# Use Docker BuildKit (faster)
DOCKER_BUILDKIT=1 docker-compose up -d --build

# Or set in environment
export DOCKER_BUILDKIT=1
```

---

## API Key Problems

### ⚠️ Warning: `HF_TOKEN: NOT SET`

**Impact:** Image generation will use fallback (solid color gradients instead of AI-generated images).

**Solution:**
```bash
# 1. Get token from https://huggingface.co/settings/tokens
# 2. Add to .env file
echo "HF_TOKEN=hf_your_token_here" >> .env

# 3. Restart Docker
docker-compose restart
```

---

### ⚠️ Warning: `No text generation API keys found`

**Impact:** Will use free Pollinations.ai fallback (works but slower).

**Solution:**
```bash
# Add at least one text generation key to .env
GOOGLE_API_KEY=AIza_your_key_here     # Recommended (free, good quality)
# OR
GROQ_API_KEY=gsk_your_key_here        # Also free, very fast
# OR
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Restart
docker-compose restart
```

---

### ❌ Error: `Invalid API key` or `401 Unauthorized`

**Cause:** API key is incorrect or expired.

**Solution:**
```bash
# 1. Verify your key is correct (check for typos)
# 2. Regenerate key from provider
# 3. Update .env file
# 4. Restart
docker-compose restart
```

---

## Generation Failures

### ❌ Error: `No ads retrieved from dataset`

**Cause:** FAISS index is empty or corrupted.

**Solution:**
```bash
# Validate FAISS index
python scripts/check_faiss.py

# If failed, re-pull from Git LFS
git lfs pull
docker-compose up -d --build
```

---

### ❌ Poor image quality (solid colors only)

**Cause:** Missing `HF_TOKEN` - using fallback image generator.

**Solution:**
```bash
# Add HuggingFace token to .env
HF_TOKEN=hf_your_token_here

# Restart
docker-compose restart

# Verify in logs
docker-compose logs | grep "HF_TOKEN"
# Should show: "HF_TOKEN: set (hf_xxxxx...)"
```

---

### ❌ Error: `CLIP model download failed`

**Cause:** No internet connection or HuggingFace is down.

**Solution:**
```bash
# Check internet
curl -I https://huggingface.co

# If HuggingFace is down, wait and retry
docker-compose restart

# Or use offline mode (if model already downloaded once)
# Model is cached in .cache/huggingface/
```

---

## Performance Issues

### 🐌 Slow generation (> 60 seconds)

**Expected times:**
- First request: 20-40 seconds (model loading)
- Subsequent requests: 5-15 seconds

**If consistently slow:**

1. **Check API keys:**
   ```bash
   docker-compose logs | grep "API"
   # Make sure GOOGLE_API_KEY or GROQ_API_KEY is configured
   ```

2. **Check resource usage:**
   ```bash
   docker stats
   # CPU should be < 80%, Memory < 2GB
   ```

3. **Check logs for errors:**
   ```bash
   docker-compose logs -f
   ```

---

### 🐌 Docker startup takes 5+ minutes

**Normal on first run:** Downloads CLIP model (~500 MB).

**On subsequent runs should be < 30 seconds.**

**If still slow:**
```bash
# Check what's taking time
docker-compose logs -f

# Common delays:
# - "Loading CLIP model" (5-10 min first time, < 1 sec after)
# - "Building brand index" (< 5 sec)
# - FAISS loading (< 2 sec)
```

---

## Quick Diagnostics

### Run Full Health Check:

```bash
# 1. Validate FAISS files
python scripts/check_faiss.py

# 2. Check Docker status
docker-compose ps

# 3. Check logs for errors
docker-compose logs --tail=100

# 4. Test health endpoint
curl http://localhost:8000/api/health
# Should return: {"status":"healthy"}

# 5. Test stats endpoint
curl http://localhost:8000/api/stats
# Should return JSON with dataset info

# 6. Check API documentation
# Open: http://localhost:8000/docs
```

---

## Still Having Issues?

### Collect Debug Information:

```bash
# 1. Check Git LFS status
git lfs ls-files

# 2. Check file sizes
ls -lh embeddings/faiss_indexes/

# 3. Export full logs
docker-compose logs > debug_logs.txt

# 4. Check Docker info
docker info > docker_info.txt

# 5. Check system resources
docker stats --no-stream > docker_stats.txt
```

Then share these files when reporting issues.

---

## Clean Slate (Nuclear Option)

If nothing works, completely reset:

```bash
# 1. Stop and remove everything
docker-compose down -v
docker system prune -a --volumes

# 2. Delete local changes
rm -rf .cache outputs uploads products_db

# 3. Re-pull Git LFS files
git lfs install
git lfs pull

# 4. Validate
python scripts/check_faiss.py

# 5. Rebuild from scratch
docker-compose up -d --build

# 6. Wait for first startup (5-10 min)
docker-compose logs -f
```

---

## Prevention Checklist

Before deploying or sharing:

- [ ] ✅ Run `git lfs install`
- [ ] ✅ Run `git lfs pull` after clone
- [ ] ✅ Run `python scripts/check_faiss.py` - should pass
- [ ] ✅ Create `.env` with required keys
- [ ] ✅ Use `docker-compose up -d --build` (not just `up`)
- [ ] ✅ Wait for CLIP model download on first run
- [ ] ✅ Test `/api/health` endpoint
- [ ] ✅ Test generation with a simple prompt

---

**Remember:** 95% of issues are caused by missing Git LFS files. Always run the validation script first!
