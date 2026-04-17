# 🚀 Hugging Face Spaces - Quick Reference

## One-Line Deployment

```bash
# Automated deployment (recommended)
./scripts/deploy-to-hf.sh https://huggingface.co/spaces/YOUR_USERNAME/madverse

# Windows
scripts\deploy-to-hf.bat https://huggingface.co/spaces/YOUR_USERNAME/madverse
```

---

## Manual Deployment (5 Steps)

### 1. Create Space
Visit: https://huggingface.co/new-space
- **Name**: madverse
- **SDK**: Docker
- **Hardware**: cpu-basic (FREE)

### 2. Clone & Copy
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/madverse
cd madverse

# Copy files (exclude: .git, .venv, data/images, outputs, uploads, products_db)
cp -r /path/to/MAdVerse/* ./
cp README_HF_SPACE.md README.md
```

### 3. Setup Git LFS
```bash
git lfs install
git lfs track "embeddings/*.pkl"
git lfs track "embeddings/faiss_indexes/*.faiss"
git lfs track "embeddings/faiss_indexes/*.pkl"
git lfs track "processed/metadata/*.csv"
git lfs track "data/annotations/*.json"
```

### 4. Push
```bash
git add .
git commit -m "Deploy MAdVerse to HF Spaces"
git push origin main
```

### 5. Add API Keys
Settings → Repository secrets:
- `GOOGLE_API_KEY` or `GROQ_API_KEY` (required)
- `HF_TOKEN` (optional, improves images)

---

## Required Files Checklist

- [x] `README.md` (use `README_HF_SPACE.md`)
- [x] `.spacesconfig.yml` - HF configuration
- [x] `Dockerfile` - Container definition
- [x] `requirements.txt` - Python dependencies
- [x] `.gitattributes` - Git LFS tracking
- [x] `app/` - Application code
- [x] `scripts/` - Startup scripts
- [x] `embeddings/` - **Git LFS tracked** (~410 MB)
- [x] `processed/` - **Git LFS tracked** (~12 MB)
- [x] `data/annotations/` - **Git LFS tracked** (~12 MB)

**Do NOT include**:
- ❌ `.git/` (use new HF repo)
- ❌ `.venv/` (not needed)
- ❌ `data/images/` (12 GB, not needed)
- ❌ `outputs/` (runtime folder)
- ❌ `uploads/` (runtime folder)
- ❌ `products_db/` (runtime folder)
- ❌ `.env` (use HF secrets)

---

## Environment Variables

### Required (at least one)
```bash
GOOGLE_API_KEY=AIza...     # Google Gemini (recommended)
GROQ_API_KEY=gsk_...       # Groq (alternative)
```

### Optional
```bash
HF_TOKEN=hf_...            # Better image quality
TOGETHER_API_KEY=...       # Alternative images
ANTHROPIC_API_KEY=sk-...   # Text fallback
```

**Add in**: Space Settings → Repository secrets

---

## Build Status

### Expected Build Time
- **First build**: 10-15 minutes
- **Subsequent**: 2-5 minutes (cached)

### Build Stages
1. ✅ Clone repo + Git LFS pull (~2 min)
2. ✅ Build Docker image (~8 min)
   - System deps (~2 min)
   - PyTorch CPU (~3 min)
   - Python packages (~2 min)
   - Copy files (~1 min)
3. ✅ Start container (~1 min)
4. ✅ Validate FAISS (~10 sec)
5. ✅ Start FastAPI (~30 sec)
6. ✅ Health check (~10 sec)

### Health Check Endpoint
```
https://YOUR_USERNAME-madverse.hf.space/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "faiss_status": "ready",
  "embeddings_count": 50000
}
```

---

## Hardware Tiers

| Tier | vCPU | RAM | Cost | For MAdVerse |
|------|------|-----|------|--------------|
| **cpu-basic** | 2 | 16 GB | FREE | ✅ **Perfect** |
| cpu-upgrade | 8 | 32 GB | $0.03/h | ⚠️ Overkill |
| t4-small | 8+T4 | 32 GB | $0.60/h | ❌ Unnecessary |

**Recommendation**: Use **cpu-basic** (free tier) - app works perfectly!

---

## Troubleshooting

### Build Fails: "FAISS index is Git LFS pointer"
```bash
# In your local repo
git lfs pull
python scripts/check_faiss.py  # Verify
git push origin main
```

### Runtime Error: "Health check failed"
```yaml
# Increase startup time in .spacesconfig.yml
startup_duration: 180
```

### Warning: "No API keys configured"
Add `GOOGLE_API_KEY` or `GROQ_API_KEY` in Space Settings

### Images are solid colors
Add `HF_TOKEN` in Space Settings

---

## Post-Deployment

### Access Your Space
```
https://YOUR_USERNAME-madverse.hf.space/
```

### View Logs
```
https://huggingface.co/spaces/YOUR_USERNAME/madverse/logs
```

### API Documentation
```
https://YOUR_USERNAME-madverse.hf.space/docs
```

### Test Features
- ✅ Home: Generate ad
- ✅ Dataset: `/dataset`
- ✅ Health: `/api/health`
- ✅ Product hub: `/hub/{id}`

---

## Monitoring

### Check Status
- Space page shows "Running" badge
- Health endpoint returns 200
- No errors in logs

### Performance
- **Idle**: ~2-3 GB RAM
- **Active**: ~4-6 GB RAM
- **Response time**: 5-15 sec per ad

### Usage Stats
View in Space page:
- Total views
- Unique visitors
- Likes & forks

---

## Updating Space

```bash
# Pull latest
git pull origin main

# Make changes
# ... edit files ...

# Push update
git add .
git commit -m "Update: description"
git push origin main

# Space rebuilds automatically
```

---

## Quick Links

- **Create Space**: https://huggingface.co/new-space
- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Git LFS**: https://git-lfs.github.com/
- **Get Gemini Key**: https://aistudio.google.com/apikey
- **Get Groq Key**: https://console.groq.com
- **Get HF Token**: https://huggingface.co/settings/tokens

---

## Support

- **Full Guide**: See `HUGGINGFACE_DEPLOYMENT.md`
- **Discord**: https://hf.co/join/discord
- **Forums**: https://discuss.huggingface.co/

---

**Total Deployment Time**: ~20 minutes (including build)

**Cost**: FREE (on cpu-basic tier) ✨
