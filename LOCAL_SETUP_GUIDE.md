# 🚀 MAdVerse - Local Setup Guide (Docker)
## Complete Setup Instructions After Cloning

---

## 📋 **Prerequisites**

- ✅ Docker Desktop installed (Windows/Mac/Linux)
- ✅ Git installed
- ✅ Internet connection

---

## 🎯 **Quick Start (5 Steps)**

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/MAdVerse.git
cd MAdVerse
```

---

### **Step 2: Create `.env` File**

The `.env` file goes in the **root directory** (same level as `docker-compose.yml`).

```bash
# Copy the example file
copy .env.example .env

# Or create manually
notepad .env
```

**Location:**
```
MAdVerse/
├── .env                    ← Create this file here!
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── app/
```

---

### **Step 3: Add API Keys to `.env`**

Open `.env` and add your keys:

```env
# Required for best image generation
HF_TOKEN=hf_your_token_here

# Required for AI text generation (at least one)
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional but recommended
TOGETHER_API_KEY=your_together_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Where to get these keys?** See Step 4 below.

---

### **Step 4: Get Free API Keys**

#### **1. HuggingFace Token (HF_TOKEN)** - For Image Generation

1. Go to: https://huggingface.co/settings/tokens
2. Sign up (free)
3. Click **"New token"**
4. Name: `MAdVerse`
5. Type: **Read**
6. Click **"Generate"**
7. Copy token (starts with `hf_`)
8. Add to `.env`: `HF_TOKEN=hf_xxxxxxxxxxxxx`

#### **2. Google Gemini API Key** - For Text Generation

1. Go to: https://aistudio.google.com/apikey
2. Sign in with Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza`)
5. Add to `.env`: `GOOGLE_API_KEY=AIzaxxxxxxxxxxxxx`

#### **3. Groq API Key** - For Fast Text Generation (Optional)

1. Go to: https://console.groq.com
2. Sign up (free)
3. Go to **API Keys**
4. Click **"Create API Key"**
5. Copy the key
6. Add to `.env`: `GROQ_API_KEY=gsk_xxxxxxxxxxxxx`

#### **4. Together AI** - Alternative Image Gen (Optional)

1. Go to: https://api.together.xyz/signup
2. Sign up
3. Get API key from dashboard
4. Add to `.env`: `TOGETHER_API_KEY=xxxxxxxxxxxxx`

---

### **Step 5: Run with Docker**

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f
```

**Wait 2-3 minutes for first boot** (downloads CLIP model).

When you see:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Open in browser:** http://localhost:8000

---

## ✅ **Verify It's Working**

### **1. Check Health**
```bash
curl http://localhost:8000/api/health
```

Should return:
```json
{"status": "healthy"}
```

### **2. Check API Keys Loaded**
```bash
docker-compose logs | findstr "HF_TOKEN"
```

Should show:
```
[STARTUP] Image generation API key status: configured (HF and/or Together).
```

### **3. Test Generation**
Open http://localhost:8000/docs and try the `/api/generate` endpoint.

---

## 📂 **Your .env File Should Look Like:**

```env
# ============================================================
# MAdVerse - Environment Variables
# ============================================================

# --- Image Generation (Required for best quality) ---
HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890

# --- Text Generation (At least one required) ---
GOOGLE_API_KEY=AIzaSyAaBbCcDdEeFfGgHhIiJjKk1234567890
GROQ_API_KEY=gsk_1234567890abcdefghijklmnopqrstuvwxyz

# --- Optional ---
TOGETHER_API_KEY=1234567890abcdefghijklmnopqrstuvwxyz
ANTHROPIC_API_KEY=sk-ant-1234567890abcdefghijklmnopqrstuvwxyz
```

**Save this file as `.env` in the project root directory.**

---

## 🔍 **Troubleshooting**

### **Problem: "HF_TOKEN: NOT SET"**

**Solution:**
```bash
# Check .env exists
dir .env

# Check .env has content
type .env

# Restart Docker
docker-compose down
docker-compose up -d
```

### **Problem: "Poor image quality"**

**Cause:** No HF_TOKEN set

**Solution:** Add `HF_TOKEN` to `.env` and restart:
```bash
docker-compose restart
```

### **Problem: "exec format error"**

**Solution:** Line ending issue (already fixed in your Dockerfile)
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### **Problem: "Container keeps restarting"**

**Solution:** Check logs for errors
```bash
docker-compose logs
```

---

## 🛑 **Stop the Application**

```bash
docker-compose down
```

---

## 🔄 **Restart After Changes**

### **After changing .env:**
```bash
docker-compose restart
```

### **After changing code:**
```bash
docker-compose down
docker-compose up -d --build
```

---

## 📊 **What Gets Downloaded on First Run**

| Component | Size | Time | When |
|-----------|------|------|------|
| Docker base image | ~400 MB | 2-3 min | First build |
| Python packages | ~1.5 GB | 5-7 min | First build |
| CLIP model | ~500 MB | 2-5 min | First start |
| **Total first time** | **~2.4 GB** | **10-15 min** | Once |

**After first run:** Starts in 30 seconds! ⚡

---

## 🌐 **Access URLs**

Once running:

- **Main App:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health
- **Dataset Explorer:** http://localhost:8000/dataset
- **Stats:** http://localhost:8000/api/stats

---

## 📁 **Project Structure**

```
MAdVerse/
├── .env                      ← YOUR API KEYS (create this!)
├── .env.example              ← Template (don't edit)
├── docker-compose.yml        ← Docker config
├── Dockerfile                ← Docker image definition
├── requirements.txt          ← Python dependencies
├── run.py                    ← Server launcher
├── app/                      ← Application code
│   ├── main.py              ← FastAPI routes
│   ├── pipeline.py          ← AI pipeline
│   └── ...
├── embeddings/               ← Pre-computed embeddings
├── data/                     ← Dataset annotations
├── outputs/                  ← Generated ads (created at runtime)
├── uploads/                  ← User uploads (created at runtime)
└── products_db/              ← SQLite database (created at runtime)
```

---

## 🎯 **Complete Setup Checklist**

- [ ] Clone repository
- [ ] Create `.env` file in root directory
- [ ] Get HF_TOKEN from HuggingFace
- [ ] Get GOOGLE_API_KEY from Google AI Studio
- [ ] Add keys to `.env` file
- [ ] Run `docker-compose up -d`
- [ ] Wait 5-10 minutes (first time only)
- [ ] Check logs: `docker-compose logs -f`
- [ ] Open http://localhost:8000
- [ ] Test generation in UI or /docs

---

## 🚀 **Summary - Commands to Run**

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/MAdVerse.git
cd MAdVerse

# 2. Create .env
copy .env.example .env
notepad .env
# Add your API keys and save

# 3. Start Docker
docker-compose up -d

# 4. Wait and watch logs
docker-compose logs -f

# 5. Access when ready
# http://localhost:8000
```

---

## 💡 **Why You Need API Keys**

| Key | Purpose | Free Tier | Required? |
|-----|---------|-----------|-----------|
| **HF_TOKEN** | High-quality image generation (FLUX.1) | ✅ Yes | Highly recommended |
| **GOOGLE_API_KEY** | AI text generation (Gemini) | ✅ Yes | At least one text gen key |
| **GROQ_API_KEY** | Fast text generation | ✅ Yes | Optional |
| **TOGETHER_API_KEY** | Alternative image gen | ✅ Trial | Optional |
| **ANTHROPIC_API_KEY** | Claude text gen | ⚠️ Limited | Optional |

**Minimum setup:** `HF_TOKEN` + `GOOGLE_API_KEY`

---

## 📝 **Notes**

1. **Never commit `.env` to Git** - it contains secrets!
2. `.env` is in `.gitignore` - safe by default ✅
3. Use `.env.example` as a template
4. All keys are FREE tier available
5. Dataset download is SKIPPED by default (uses embeddings only)

---

## 🆘 **Need Help?**

- Health endpoint not responding → Check logs
- Poor image quality → Add HF_TOKEN
- No text generation → Add GOOGLE_API_KEY or GROQ_API_KEY
- Container crashes → Check `docker-compose logs`

---

**Ready to deploy? See DEPLOYMENT_GUIDE.md for Hugging Face Spaces deployment!** 🚀
