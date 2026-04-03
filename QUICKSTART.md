# ⚡ QUICK START - Copy & Paste Commands

## 1️⃣ Install Git LFS (One-Time)
```bash
# Windows: Download from https://git-lfs.github.com/ then run:
git lfs install

# Linux:
sudo apt-get install git-lfs && git lfs install

# Mac:
brew install git-lfs && git lfs install
```

---

## 2️⃣ Clone & Download Files
```bash
git clone https://github.com/YOUR_USERNAME/MAdVerse.git
cd MAdVerse
git lfs pull
python scripts/check_faiss.py
```

✅ **Validation should show:** `"✓ ALL CHECKS PASSED!"`

---

## 3️⃣ Setup Environment
```bash
# Copy template
copy .env.example .env

# Edit and add keys (Windows)
notepad .env

# Or (Linux/Mac)
nano .env
```

**Add these lines:**
```
HF_TOKEN=hf_your_token_here
GOOGLE_API_KEY=AIza_your_key_here
```

**Get keys:**
- HF: https://huggingface.co/settings/tokens
- Google: https://aistudio.google.com/apikey

---

## 4️⃣ Start Docker
```bash
docker-compose up -d --build
docker-compose logs -f
```

⏳ **Wait 5-10 min on first run** (downloads CLIP model)

Look for: `"INFO: Uvicorn running on http://0.0.0.0:8000"`

Press **Ctrl+C** to exit logs

---

## 5️⃣ Access App
Open browser: **http://localhost:8000**

---

## 🔍 Verify
```bash
curl http://localhost:8000/api/health
```
Should return: `{"status":"healthy"}`

---

## 🆘 If Error "FAISS index error":
```bash
git lfs install
git lfs pull
docker-compose down
docker-compose up -d --build
```

---

## 📚 Full guides:
- [COMPLETE_SETUP_COMMANDS.md](COMPLETE_SETUP_COMMANDS.md) - Detailed commands
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - If issues occur
- [QUICK_FIX_FAISS.md](QUICK_FIX_FAISS.md) - FAISS error fix
