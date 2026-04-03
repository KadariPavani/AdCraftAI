# 🚀 Quick Deployment Guide - HF Space with Cloudinary

## ⚡ TL;DR - Deploy in 3 Steps

### Step 1: Configure HF Space Secrets (5 minutes)

Go to: **Your HF Space → Settings → Repository secrets**

**Add these secrets** (get values from your local `.env` file):

```bash
# Storage (REQUIRED for persistent images)
USE_CLOUD_STORAGE=true
CLOUDINARY_CLOUD_NAME=<from_your_env_file>
CLOUDINARY_API_KEY=<from_your_env_file>
CLOUDINARY_API_SECRET=<from_your_env_file>

# Text Generation (REQUIRED - at least one)
GOOGLE_API_KEY=<from_your_env_file>
GROQ_API_KEY=<from_your_env_file>

# Optional (better quality)
HF_TOKEN=<from_your_env_file>
TOGETHER_API_KEY=<from_your_env_file>
ANTHROPIC_API_KEY=<from_your_env_file>
```

**⚠️ NEVER commit actual API keys! Only add them to HF Space settings.**

### Step 2: Push Code (1 minute)

```bash
cd P:\KHUB\madverse
git add .
git commit -m "feat: Integrate Cloudinary cloud storage

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push
```

### Step 3: Verify (2 minutes)

1. Wait for HF Space to rebuild (~5-10 minutes)
2. Check build logs for: `[CLOUDINARY] ✅ Configured: dnjaydxsi`
3. Open your space and generate a test ad
4. Verify image appears and persists

---

## ✅ What You Get

- ✅ **Persistent images** - survive container restarts
- ✅ **CDN delivery** - fast loading worldwide
- ✅ **Product hub works** - shareable links never break
- ✅ **Auto-fallback** - uses local if Cloudinary fails
- ✅ **Free tier** - 25GB storage included

---

## 🔍 Verify Success

**In HF Space build logs, you should see:**
```
[CLOUDINARY] ✅ Configured: dnjaydxsi
[STORAGE] ✅ Using Cloudinary
[INIT] Storage initialized
```

**Generate a test ad and check:**
- Image loads correctly
- Refresh page - image still there
- Visit Cloudinary dashboard - see image in `madverse/outputs/`

---

## 🆘 Quick Troubleshooting

**If images are local instead of Cloudinary:**
- Check: `USE_CLOUD_STORAGE=true` (lowercase "true")
- Verify all 4 Cloudinary secrets are set
- Restart HF Space

**If build fails:**
- Check: `cloudinary` is in `requirements.txt` (it is ✅)
- Check: All secrets have correct names (no typos)

---

## 📊 Current Setup Status

- ✅ Code integrated (pipeline.py + main.py)
- ✅ Storage modules ready (app/storage/)
- ✅ Documentation updated
- ✅ .env.example created
- ✅ Requirements.txt has cloudinary
- ✅ .gitignore excludes .env

**Status: READY TO DEPLOY** 🚀

---

For detailed information, see: `CLOUDINARY_INTEGRATION_COMPLETE.md`
