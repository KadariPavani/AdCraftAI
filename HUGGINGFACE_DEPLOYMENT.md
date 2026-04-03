# 🚀 Hugging Face Spaces Deployment Guide

Complete guide to deploy **MAdVerse (AdCraft AI)** to Hugging Face Spaces.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Deployment](#quick-deployment)
3. [Detailed Step-by-Step](#detailed-step-by-step)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### ✅ Before You Start

- [ ] **Hugging Face Account**: [Sign up free](https://huggingface.co/join)
- [ ] **Git LFS Installed**: Required for large files
  ```bash
  git lfs install
  ```
- [ ] **API Keys** (at least one recommended):
  - **Google Gemini**: [Get free key](https://aistudio.google.com/apikey)
  - **Groq**: [Get free key](https://console.groq.com)
  - **HF Token**: [Get free token](https://huggingface.co/settings/tokens) (for images)

### 📦 What Gets Deployed

| Component | Size | Required | Notes |
|-----------|------|----------|-------|
| Application code | ~5 MB | ✅ Yes | FastAPI app + Python modules |
| Docker configuration | ~1 KB | ✅ Yes | Dockerfile + docker-compose.yml |
| FAISS index | ~345 MB | ✅ Yes | Pre-computed embeddings (Git LFS) |
| Embeddings | ~65 MB | ✅ Yes | CLIP vectors (Git LFS) |
| Metadata | ~12 MB | ✅ Yes | Processed annotations (Git LFS) |
| Annotations | ~12 MB | ✅ Yes | Dataset metadata (Git LFS) |
| Dataset images | ~12 GB | ❌ No | Not deployed (embeddings only) |

**Total deployed size**: ~450 MB (within HF Spaces free tier limits ✅)

---

## Quick Deployment

### 🎯 Option 1: Deploy from GitHub (Recommended)

**If your code is already on GitHub:**

1. Go to [Hugging Face Spaces](https://huggingface.co/new-space)
2. Click **"Create new Space"**
3. Fill in details:
   - **Name**: `madverse` (or your choice)
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: cpu-basic (free)
4. Click **"Import from GitHub"**
5. Enter your GitHub repo URL
6. Click **"Create Space"**
7. Go to **Settings → Repository secrets**
8. Add environment variables:
   ```
   GOOGLE_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   HF_TOKEN=your_token_here
   ```
9. Wait 10-15 minutes for build
10. Done! 🎉

### 🎯 Option 2: Direct Git Push

**If deploying directly to HF:**

```bash
# 1. Create Space on Hugging Face
# Visit: https://huggingface.co/new-space
# Name: madverse, SDK: Docker, Hardware: cpu-basic

# 2. Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/madverse
cd madverse

# 3. Copy your project files
# Copy everything from your MAdVerse project EXCEPT:
# - .git/ (will use new HF repo)
# - .venv/ (not needed)
# - data/images/ (too large, not needed)
# - outputs/ (runtime folder)
# - uploads/ (runtime folder)
# - products_db/ (runtime folder)

# 4. Ensure Git LFS is tracking large files
git lfs track "embeddings/image_embeddings.pkl"
git lfs track "embeddings/faiss_indexes/madverse_index.faiss"
git lfs track "embeddings/faiss_indexes/id_to_metadata.pkl"
git lfs track "processed/metadata/madverse_metadata.csv"
git lfs track "data/annotations/web_annot_j.json"

# 5. Copy the HF Space README
cp README_HF_SPACE.md README.md

# 6. Commit and push
git add .
git commit -m "Initial deployment to Hugging Face Spaces"
git push origin main

# 7. Add secrets in HF Space settings
# Visit: https://huggingface.co/spaces/YOUR_USERNAME/madverse/settings
# Add: GOOGLE_API_KEY, GROQ_API_KEY, HF_TOKEN
```

---

## Detailed Step-by-Step

### Step 1: Create Hugging Face Space

1. **Login to Hugging Face**: [huggingface.co](https://huggingface.co)

2. **Create New Space**:
   - Click your profile → "New Space"
   - Or visit: [huggingface.co/new-space](https://huggingface.co/new-space)

3. **Configure Space**:
   ```
   Space name: madverse (or your choice)
   License: MIT
   Visibility: Public (or Private)
   SDK: Docker
   Hardware: cpu-basic (FREE tier)
   Space template: Blank
   ```

4. **Click "Create Space"**

### Step 2: Prepare Your Repository

#### 2A. Verify Git LFS Setup

```bash
# Check if Git LFS is installed
git lfs version

# If not installed, install it:
# Windows: Download from https://git-lfs.github.com/
# Mac: brew install git-lfs
# Linux: sudo apt-get install git-lfs

# Initialize Git LFS in your repo
git lfs install
```

#### 2B. Verify LFS Files Are Tracked

```bash
# Check .gitattributes file exists and contains:
cat .gitattributes
```

Should show:
```
embeddings/image_embeddings.pkl filter=lfs diff=lfs merge=lfs -text
embeddings/faiss_indexes/madverse_index.faiss filter=lfs diff=lfs merge=lfs -text
processed/metadata/madverse_metadata.csv filter=lfs diff=lfs merge=lfs -text
embeddings/faiss_indexes/id_to_metadata.pkl filter=lfs diff=lfs merge=lfs -text
data/annotations/web_annot_j.json filter=lfs diff=lfs merge=lfs -text
```

#### 2C. Verify LFS Files Are Downloaded

```bash
# Pull all LFS files
git lfs pull

# Verify FAISS index is not a pointer
python scripts/check_faiss.py
```

### Step 3: Prepare Files for Deployment

#### 3A. Copy HF Space README

The `README_HF_SPACE.md` file will become your Space's main README:

```bash
# Backup original README (optional)
cp README.md README_GITHUB.md

# Use HF Space README
cp README_HF_SPACE.md README.md
```

#### 3B. Verify Configuration Files Exist

Check these files are present:
- ✅ `.spacesconfig.yml` - Hugging Face Space configuration
- ✅ `Dockerfile` - Container definition
- ✅ `requirements.txt` - Python dependencies
- ✅ `docker-compose.yml` - Local Docker config (optional, not used on HF)
- ✅ `scripts/start.sh` - Startup script
- ✅ `scripts/docker-entrypoint.sh` - Validation script
- ✅ `.env.example` - Environment variable template

#### 3C. Verify Dockerfile Port Configuration

The Dockerfile should expose port 7860 (HF Spaces default):

```dockerfile
# Line 83-84 in Dockerfile
EXPOSE 7860 8000
```

And `start.sh` should use `${PORT:-7860}`:

```bash
# Line 113-114 in scripts/start.sh
exec python -m uvicorn app.main:app \
    --port "${PORT:-7860}"
```

### Step 4: Push to Hugging Face

#### 4A. Clone Your HF Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/madverse
cd madverse
```

#### 4B. Copy Project Files

**On Windows:**
```powershell
# Copy all files except excluded folders
robocopy P:\KHUB\product-labs\MAdVerse . /E /XD .git .venv data\images outputs uploads products_db __pycache__ .pytest_cache

# Or manually copy:
# - app/
# - scripts/
# - embeddings/
# - processed/
# - data/annotations/
# - data/indices/
# - All .py files (run.py, DatasetLoad.py, etc.)
# - All .md files
# - All config files (Dockerfile, requirements.txt, etc.)
```

**On Linux/Mac:**
```bash
rsync -av --exclude='.git' --exclude='.venv' --exclude='data/images' \
  --exclude='outputs' --exclude='uploads' --exclude='products_db' \
  /path/to/MAdVerse/ ./
```

#### 4C. Ensure Git LFS Tracking

```bash
# Add .gitattributes (should already exist)
git add .gitattributes

# Track large files
git lfs track "embeddings/image_embeddings.pkl"
git lfs track "embeddings/faiss_indexes/*.faiss"
git lfs track "embeddings/faiss_indexes/*.pkl"
git lfs track "processed/metadata/*.csv"
git lfs track "data/annotations/*.json"

# Verify LFS tracking
git lfs ls-files
```

#### 4D. Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Deploy MAdVerse AI to Hugging Face Spaces

- FastAPI ad generation platform
- FAISS semantic search (50K+ ads)
- Multi-language support (50+ languages)
- Docker-based deployment
- Free tier compatible (cpu-basic)
"

# Push to HF (may take 5-10 minutes for large files)
git push origin main

# Monitor upload progress
# Large LFS files will show upload percentage
```

### Step 5: Configure Environment Variables

1. **Go to Space Settings**:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/madverse/settings
   ```

2. **Scroll to "Repository secrets"**

3. **Add Required Secrets**:

   | Variable | Description | Required | Get It Here |
   |----------|-------------|----------|-------------|
   | `GOOGLE_API_KEY` | Google Gemini API key | Recommended | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
   | `GROQ_API_KEY` | Groq API key | Recommended | [console.groq.com](https://console.groq.com) |
   | `HF_TOKEN` | Hugging Face token | Optional | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
   | `TOGETHER_API_KEY` | Together AI key | Optional | [api.together.xyz](https://api.together.xyz) |
   | `ANTHROPIC_API_KEY` | Claude API key | Optional | [console.anthropic.com](https://console.anthropic.com) |

4. **Click "Add"** for each secret

5. **Space will automatically rebuild** after adding secrets

### Step 6: Monitor Build

1. **View Build Logs**:
   - Go to your Space page
   - Click "Logs" tab
   - Watch the build progress

2. **Expected Build Stages**:
   ```
   ✅ Cloning repository
   ✅ Downloading Git LFS files
   ✅ Building Docker image
      - Installing system dependencies (~2 min)
      - Installing PyTorch CPU (~3 min)
      - Installing Python packages (~2 min)
      - Copying application files (~1 min)
   ✅ Starting container
   ✅ Running entrypoint validation
      - Checking FAISS index
      - Checking metadata files
   ✅ Starting FastAPI server
   ✅ Running health checks
   ✅ Space is ready!
   ```

3. **Build Duration**: 10-15 minutes (first time)

4. **Subsequent builds**: ~2-5 minutes (cached layers)

### Step 7: Verify Deployment

1. **Check Space is Running**:
   - Status badge should show "Running"
   - Not "Building" or "Error"

2. **Test the Application**:
   ```
   https://YOUR_USERNAME-madverse.hf.space
   ```

3. **Test API Endpoint**:
   ```
   https://YOUR_USERNAME-madverse.hf.space/api/health
   ```

   Should return:
   ```json
   {
     "status": "healthy",
     "timestamp": "2024-...",
     "faiss_status": "ready",
     "embeddings_count": 50000
   }
   ```

4. **Test Main Features**:
   - ✅ Home page loads
   - ✅ Dataset explorer works (`/dataset`)
   - ✅ Can parse prompt
   - ✅ Can generate ad
   - ✅ Images generate (if HF_TOKEN set)
   - ✅ Multi-language captions work

---

## Configuration

### Environment Variables

Set in **Space Settings → Repository secrets**:

#### Required (at least one):

```bash
# Google Gemini (recommended - fast, reliable, generous free tier)
GOOGLE_API_KEY=AIza...

# Groq (alternative - very fast LPU inference)
GROQ_API_KEY=gsk_...
```

**Without any text generation API**: App uses free Pollinations.ai (slower, lower quality)

#### Optional:

```bash
# Hugging Face (for better image quality)
HF_TOKEN=hf_...

# Together AI (alternative image generation)
TOGETHER_API_KEY=...

# Anthropic Claude (text generation fallback)
ANTHROPIC_API_KEY=sk-ant-...
```

### Hardware Selection

| Tier | vCPU | RAM | GPU | Cost | Suitable? |
|------|------|-----|-----|------|-----------|
| **cpu-basic** | 2 | 16 GB | No | FREE | ✅ **Recommended** |
| cpu-upgrade | 8 | 32 GB | No | $0.03/h | ⚠️ Overkill |
| t4-small | 8 | 32 GB | T4 | $0.60/h | ❌ Not needed |
| t4-medium | 16 | 64 GB | T4 | $1.05/h | ❌ Not needed |

**Why cpu-basic is perfect:**
- FAISS index fits in 16 GB RAM (~350 MB + models ~2 GB)
- Image generation uses external APIs (FLUX.1, Pollinations)
- Text generation uses external APIs (Gemini, Groq)
- No GPU computation needed locally

### Space Configuration

Edit `.spacesconfig.yml` if needed:

```yaml
sdk: docker              # Don't change
app_port: 7860          # Don't change (HF default)
hardware: cpu-basic     # Change to upgrade if needed
startup_duration: 120   # Increase if startup timeout occurs
enable_internet: true   # Required for APIs
```

### Docker Configuration

The `Dockerfile` is already optimized for HF Spaces:

- ✅ Uses Python 3.11 slim base
- ✅ Installs PyTorch CPU-only (small, fast)
- ✅ Exposes port 7860
- ✅ Includes health check
- ✅ SKIP_DATASET_DOWNLOAD=1 (uses embeddings only)
- ✅ Proper cache directories for HF models

**No changes needed** unless you want to:
- Use different Python version (change line 6)
- Adjust memory/CPU limits (edit `.spacesconfig.yml`)
- Add custom fonts (add to apt-get install, line 18-31)

---

## Troubleshooting

### Build Failures

#### Error: "Git LFS files not downloaded"

**Symptom**: Build fails with "FAISS index is a Git LFS pointer"

**Cause**: Git LFS files not uploaded

**Fix**:
```bash
# In your local repo
git lfs install
git lfs pull

# Verify files are real (not pointers)
python scripts/check_faiss.py

# Re-push
git push origin main
```

#### Error: "Dockerfile build timeout"

**Symptom**: Build stops after 30 minutes

**Cause**: Default timeout too short

**Fix**:
```yaml
# Edit .spacesconfig.yml
build_timeout: 1800  # 30 minutes
```

#### Error: "Out of memory during build"

**Symptom**: Build crashes at "Installing PyTorch"

**Cause**: Insufficient memory (rare on HF)

**Fix**:
```yaml
# Upgrade to cpu-upgrade temporarily
hardware: cpu-upgrade

# After successful build, downgrade back
hardware: cpu-basic
```

### Runtime Failures

#### Error: "Application startup failed"

**Symptom**: Space shows "Error" status

**Check**:
1. View logs for error details
2. Verify FAISS index loaded:
   ```
   [STARTUP] ✓ FAISS index OK (345 MB)
   ```

**Fix**:
```bash
# If FAISS error, re-pull LFS files
git lfs pull
git push origin main
```

#### Error: "Health check failed"

**Symptom**: Space restarts repeatedly

**Check logs for**:
```
HEALTHCHECK: connection refused
```

**Fix**:
```yaml
# Increase startup duration
startup_duration: 180  # 3 minutes
```

#### Error: "No text generation API keys"

**Symptom**: Warning in logs:
```
⚠  WARNING: No text generation API keys found
```

**Impact**: Uses free Pollinations.ai (slower)

**Fix**: Add `GOOGLE_API_KEY` or `GROQ_API_KEY` in Space settings

#### Error: "Image generation returns solid colors"

**Symptom**: Generated images are plain gradients

**Cause**: No `HF_TOKEN` configured

**Fix**: Add `HF_TOKEN` in Space settings

### Performance Issues

#### Slow ad generation

**Possible causes**:
1. Using free APIs (Pollinations) without paid API keys
2. First run (downloading CLIP model ~1 GB)
3. Cold start after Space sleep

**Fix**:
- Add `GOOGLE_API_KEY` (Gemini is very fast)
- Add `HF_TOKEN` (FLUX is faster than Pollinations)
- Disable Space sleep: `disable_sleep: true` in `.spacesconfig.yml`

#### High memory usage

**Check logs**:
```
docker stats
```

**Normal usage**:
- Idle: ~2-3 GB
- Generating: ~4-6 GB
- Peak: ~8 GB

**If > 12 GB**: Upgrade to `cpu-upgrade`

### Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| **LFS pointer uploaded** | "FAISS index too small" | `git lfs pull` then push |
| **Wrong port** | Can't access Space | Ensure Dockerfile uses `7860` |
| **Missing env vars** | Low quality output | Add API keys in Settings |
| **Slow startup** | Timeout errors | Increase `startup_duration` |
| **Space sleeping** | First request slow | Set `disable_sleep: true` |

---

## Monitoring & Maintenance

### View Logs

**Real-time logs**:
```
https://huggingface.co/spaces/YOUR_USERNAME/madverse/logs
```

**Download logs**:
- Click "Logs" tab
- Click "Download logs"

### Monitor Usage

**Analytics** (if public Space):
- Views, likes, forks
- Space status uptime

**Resource usage**:
- View in logs: Memory, CPU percentage
- Check health endpoint: `/api/health`

### Update Deployment

**Minor updates** (code only):
```bash
# Pull latest from HF
git pull origin main

# Make changes
# ... edit files ...

# Push updates
git add .
git commit -m "Update: describe changes"
git push origin main

# Space automatically rebuilds
```

**Major updates** (dependencies, Dockerfile):
```bash
# Update requirements.txt or Dockerfile
# ... make changes ...

# Push (will trigger full rebuild)
git push origin main

# Monitor build in Logs tab
```

### Backup

**Export Space**:
1. Clone Space locally:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/madverse
   ```

2. Backup database (if needed):
   - Download from Space persistent storage (if enabled)

3. Backup environment variables:
   - Copy from Settings → Repository secrets

### Rollback

**If deployment breaks**:

1. **Find last working commit**:
   ```bash
   git log --oneline
   ```

2. **Revert to that commit**:
   ```bash
   git revert <commit-hash>
   git push origin main
   ```

3. **Or force push previous version**:
   ```bash
   git reset --hard <commit-hash>
   git push origin main --force
   ```

---

## Advanced Configuration

### Enable Persistent Storage

**For saving products across restarts**:

1. **Go to Space Settings**
2. **Enable "Persistent storage"**
3. **Set size**: 10 GB (for products_db)
4. **Mount path**: `/app/products_db`

**Cost**: ~$5/month for 10 GB

### Custom Domain

**Map your domain to Space**:

1. **Settings → "Custom domain"**
2. **Enter domain**: `ads.yourdomain.com`
3. **Add CNAME record** in your DNS:
   ```
   CNAME: ads.yourdomain.com → YOUR_USERNAME-madverse.hf.space
   ```

### Private Space

**Make Space private**:

1. **Settings → "Visibility"**
2. **Select "Private"**
3. **Space accessible only to you**

**Cost**: Free for private Spaces

### Scheduled Restarts

**Auto-restart weekly** (clears cache, updates):

1. **Settings → "Advanced"**
2. **Enable "Scheduled restart"**
3. **Set schedule**: Weekly, Sunday 3 AM

---

## Deployment Checklist

### Pre-Deployment

- [ ] Git LFS installed and configured
- [ ] All LFS files downloaded (`git lfs pull`)
- [ ] FAISS index validated (`python scripts/check_faiss.py`)
- [ ] API keys obtained (Google, Groq, HF)
- [ ] README_HF_SPACE.md reviewed and copied to README.md
- [ ] .spacesconfig.yml exists
- [ ] Dockerfile exposes port 7860
- [ ] No .env file in repository (use HF secrets instead)

### During Deployment

- [ ] Space created on Hugging Face
- [ ] Repository cloned locally
- [ ] Files copied (excluding .git, .venv, data/images, outputs, uploads)
- [ ] .gitattributes configured for LFS
- [ ] All files committed and pushed
- [ ] Environment variables added in Space settings
- [ ] Build logs monitored (no errors)

### Post-Deployment

- [ ] Space status: "Running" (not "Error")
- [ ] Health endpoint returns 200: `/api/health`
- [ ] Home page loads successfully: `/`
- [ ] Dataset explorer works: `/dataset`
- [ ] Can generate test ad
- [ ] Images generate properly (if HF_TOKEN set)
- [ ] Multi-language captions work
- [ ] API documentation accessible: `/docs`
- [ ] Logs show no errors

### Optional

- [ ] Persistent storage enabled
- [ ] Custom domain configured
- [ ] Analytics tracking setup
- [ ] Backup strategy defined
- [ ] Monitoring alerts configured

---

## Support & Resources

### Documentation

- **Hugging Face Spaces**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Docker SDK**: [huggingface.co/docs/hub/spaces-sdks-docker](https://huggingface.co/docs/hub/spaces-sdks-docker)
- **Git LFS**: [git-lfs.github.com](https://git-lfs.github.com/)

### Community

- **HF Discord**: [hf.co/join/discord](https://hf.co/join/discord)
- **HF Forums**: [discuss.huggingface.co](https://discuss.huggingface.co/)

### API Keys

- **Google Gemini**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Groq**: [console.groq.com](https://console.groq.com)
- **Hugging Face**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## Next Steps

After successful deployment:

1. **Share your Space**: Get feedback from users
2. **Monitor performance**: Check logs regularly
3. **Iterate**: Improve based on user feedback
4. **Scale**: Upgrade hardware if needed
5. **Showcase**: Add to HF Collections, share on social media

---

**🎉 Congratulations on deploying MAdVerse to Hugging Face Spaces!**

If you found this guide helpful, please ⭐ star the repository!

---

**Last Updated**: April 2026  
**Guide Version**: 1.0  
**Tested on**: Hugging Face Spaces (Docker SDK, cpu-basic tier)
