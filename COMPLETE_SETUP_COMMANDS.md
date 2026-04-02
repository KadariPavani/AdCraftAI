# 🚀 MAdVerse - Complete Setup Commands

## Fresh Clone to Running App in 10 Commands

Copy and paste these commands in sequence:

---

## **Step 1: Install Git LFS (One-Time Setup)**

### Windows:
1. Download installer from: https://git-lfs.github.com/
2. Run installer
3. Open terminal and run:
```bash
git lfs install
```

### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```

### Mac:
```bash
brew install git-lfs
git lfs install
```

---

## **Step 2: Clone and Setup**

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/MAdVerse.git

# 2. Enter directory
cd MAdVerse

# 3. Pull Git LFS files (CRITICAL!)
git lfs pull

# 4. Validate files downloaded correctly
python scripts/check_faiss.py

# 5. Create .env file from template
copy .env.example .env
# On Linux/Mac use: cp .env.example .env
```

---

## **Step 3: Add API Keys to .env**

Edit the `.env` file and add your keys:

```bash
# Windows
notepad .env

# Linux/Mac
nano .env
# or
vim .env
```

**Add these lines to .env:**
```env
# Required for best image quality
HF_TOKEN=hf_your_token_here

# Required for text generation (at least one)
GOOGLE_API_KEY=AIza_your_key_here
GROQ_API_KEY=gsk_your_key_here

# Optional
TOGETHER_API_KEY=your_key_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

**Get FREE API keys:**
- HuggingFace Token: https://huggingface.co/settings/tokens
- Google Gemini API: https://aistudio.google.com/apikey
- Groq API: https://console.groq.com

---

## **Step 4: Run with Docker**

```bash
# 6. Start Docker (build + run)
docker-compose up -d --build

# 7. Watch logs (wait 5-10 min on first run)
docker-compose logs -f

# Press Ctrl+C to exit logs when you see:
# "INFO: Uvicorn running on http://0.0.0.0:8000"
```

---

## **Step 5: Verify It's Working**

```bash
# 8. Check health endpoint
curl http://localhost:8000/api/health

# Should return: {"status":"healthy"}

# 9. Check stats
curl http://localhost:8000/api/stats
```

---

## **Step 6: Access the App**

Open in browser:
- **Main App:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Dataset Explorer:** http://localhost:8000/dataset

---

## **Complete Command List (Copy-Paste)**

```bash
# === SETUP (One time) ===
git lfs install
git clone https://github.com/YOUR_USERNAME/MAdVerse.git
cd MAdVerse
git lfs pull
python scripts/check_faiss.py

# === CONFIGURE ===
copy .env.example .env
# Edit .env and add your API keys
notepad .env

# === RUN ===
docker-compose up -d --build
docker-compose logs -f

# === VERIFY ===
curl http://localhost:8000/api/health
```

Then open: **http://localhost:8000**

---

## **Alternative: Run Without Docker (Local Python)**

If you prefer to run without Docker:

```bash
# 1-5: Same as above (clone, git lfs pull, create .env)

# 6. Create virtual environment
python -m venv .venv

# 7. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 8. Install dependencies
pip install -r requirements.txt

# 9. Run the app
python run.py

# 10. Open browser
# http://localhost:8000
```

---

## **Common Commands**

### Start App:
```bash
docker-compose up -d
```

### Stop App:
```bash
docker-compose down
```

### View Logs:
```bash
docker-compose logs -f
```

### Restart After .env Changes:
```bash
docker-compose restart
```

### Rebuild After Code Changes:
```bash
docker-compose down
docker-compose up -d --build
```

### Check Status:
```bash
docker-compose ps
```

---

## **Troubleshooting**

### Error: "FAISS index error"
```bash
git lfs install
git lfs pull
python scripts/check_faiss.py
docker-compose up -d --build
```

### Error: "Port 8000 already in use"
```bash
# Check what's using the port
# Windows:
netstat -ano | findstr :8000
# Linux/Mac:
lsof -i :8000

# Change port in docker-compose.yml:
ports:
  - "8001:8000"  # Use 8001 instead
```

### Error: "Container keeps restarting"
```bash
# Check logs for errors
docker-compose logs

# Common fix: rebuild
docker-compose down
docker-compose up -d --build
```

### Validation Failed
```bash
# Run validation
python scripts/check_faiss.py

# If it shows "Git LFS pointer", pull files:
git lfs fetch --all
git lfs pull
```

---

## **Complete First-Time Setup (All Commands)**

```bash
# === INSTALL GIT LFS (Windows - download installer first) ===
git lfs install

# === CLONE PROJECT ===
git clone https://github.com/YOUR_USERNAME/MAdVerse.git
cd MAdVerse

# === DOWNLOAD LARGE FILES ===
git lfs pull

# === VALIDATE ===
python scripts/check_faiss.py
# Should show: "✓ ALL CHECKS PASSED!"

# === CONFIGURE ENVIRONMENT ===
copy .env.example .env
# Windows: notepad .env
# Linux/Mac: nano .env

# Add to .env:
# HF_TOKEN=hf_xxxxxxxxxxxxx
# GOOGLE_API_KEY=AIzaxxxxxxxxxxxxx

# === BUILD AND RUN ===
docker-compose up -d --build

# === WATCH STARTUP (wait 5-10 min first time) ===
docker-compose logs -f
# Look for: "INFO: Uvicorn running on http://0.0.0.0:8000"
# Press Ctrl+C to exit

# === VERIFY ===
curl http://localhost:8000/api/health

# === OPEN IN BROWSER ===
# http://localhost:8000
```

---

## **Quick Reference**

| Task | Command |
|------|---------|
| Start | `docker-compose up -d` |
| Stop | `docker-compose down` |
| Logs | `docker-compose logs -f` |
| Restart | `docker-compose restart` |
| Rebuild | `docker-compose up -d --build` |
| Status | `docker-compose ps` |
| Health Check | `curl http://localhost:8000/api/health` |
| Validate Setup | `python scripts/check_faiss.py` |

---

## **Expected First Startup Times**

- **First run:** 10-15 minutes (downloads CLIP model ~500 MB)
- **Subsequent runs:** 30-60 seconds

---

## **Need Help?**

- Quick fix: [QUICK_FIX_FAISS.md](QUICK_FIX_FAISS.md)
- Full guide: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Setup guide: [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)

---

**That's it! You're ready to generate ads! 🎉**
