# 🚀 MAdVerse Docker - Quick Reference

## ⚡ Quick Start (3 Steps)

```bash
# 1. Setup .env file with your API keys
copy .env.example .env
notepad .env  # Add HF_TOKEN, GOOGLE_API_KEY, etc.

# 2. Start with Docker Compose
docker-compose up -d

# 3. Access
# http://localhost:8000
```

---

## 📋 Common Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart (after changing .env)
docker-compose restart

# View logs
docker-compose logs -f

# View live logs for specific service
docker-compose logs -f madverse

# Rebuild after code changes
docker-compose up -d --build

# Check status
docker-compose ps

# Execute command in container
docker-compose exec madverse bash
```

---

## 🔍 Check Everything Works

```bash
# 1. Container running?
docker ps | grep madverse

# 2. Health check
curl http://localhost:8000/api/health

# 3. Check logs for API keys
docker-compose logs | grep "HF_TOKEN"
# Should show: [INIT] HF_TOKEN: set (hf_xxxxx...)

# 4. Test image generation
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Red Bull energy drink\",\"num_variations\":1}"
```

---

## 🎯 URLs

- **App:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/api/health
- **Dataset Explorer:** http://localhost:8000/dataset

---

## 🔑 Environment Variables (.env)

```env
# Required for best image quality
HF_TOKEN=hf_your_token_here

# Required for text generation (at least one)
GOOGLE_API_KEY=your_key
GROQ_API_KEY=your_key

# Optional
TOGETHER_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

Get tokens:
- HF: https://huggingface.co/settings/tokens
- Google: https://aistudio.google.com/apikey
- Groq: https://console.groq.com

---

## 📂 Persistent Data

Automatically saved on your host:

```
./products_db/    ← SQLite database
./outputs/        ← Generated ads
./uploads/        ← Uploaded images
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Poor image quality | Add `HF_TOKEN` to `.env` and restart |
| Database resets | Check `docker-compose.yml` has volume mounts |
| Can't access | Check port 8000 is not in use: `netstat -ano \| findstr 8000` |
| Container crashes | Check logs: `docker-compose logs` |
| Out of memory | Close other apps, Docker needs 4GB+ |

---

## 🔄 Update After Changes

```bash
# After changing .env
docker-compose restart

# After changing code
docker-compose up -d --build

# After changing docker-compose.yml
docker-compose down && docker-compose up -d

# Fresh start (removes all data!)
docker-compose down -v && docker-compose up -d
```

---

## 🗑️ Cleanup

```bash
# Stop and remove container
docker-compose down

# Remove container + volumes (deletes database!)
docker-compose down -v

# Remove image
docker rmi madverse-app

# Full cleanup
docker-compose down -v --rmi all
```

---

## ⚙️ Alternative: Manual Docker Run (Windows)

```bash
docker build -t madverse-app .

docker run -d ^
  --name madverse-app ^
  -p 8000:8000 ^
  --env-file .env ^
  -v "%cd%\products_db:/app/products_db" ^
  -v "%cd%\outputs:/app/outputs" ^
  -v "%cd%\uploads:/app/uploads" ^
  madverse-app
```

---

## 📊 Monitor Resources

```bash
# Check resource usage
docker stats madverse-app

# Check disk usage
docker system df
```

---

## 🎉 Success Checklist

- [ ] `.env` file created with API keys
- [ ] `docker-compose up -d` runs without errors
- [ ] `http://localhost:8000` loads the UI
- [ ] `/api/health` returns `{"status":"healthy"}`
- [ ] Logs show `HF_TOKEN: set`
- [ ] Generated images appear in `outputs/`
- [ ] Database persists after restart

---

For detailed guide, see: **DOCKER_GUIDE.md**
