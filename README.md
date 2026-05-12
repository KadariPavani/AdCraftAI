---
title: AdkraftAI
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# MAdVerse API Backend

FastAPI backend for ad generation and AdCraft_AI integration.

## Run modes

- Local app mode: `python run.py`
- AdCraft integration mode: `python run.py --mode integration` (or `python run_for_adcraft.py`)
- Deployment mode: `python run.py --mode deploy`

## Hugging Face deployment

This project is configured for Docker Spaces.

1. Create/update Space: `PavaniKadari/AdkraftAI`
2. Push this folder to that Space repository
3. Set required secrets (`HF_TOKEN`, `GOOGLE_API_KEY`/`GROQ_API_KEY`, etc.)

Expected Space URL:

`https://pavanikadari-adkraftai.hf.space`

Health check:

`https://pavanikadari-adkraftai.hf.space/api/health`

## AdCraft_AI integration (deployed backend)

In `AdCraft_AI/.env`:

```env
VITE_MADVERSE_API_URL=https://pavanikadari-adkraftai.hf.space
VITE_MADVERSE_ENABLED=true
```

Then run only AdCraft frontend:

```bash
npm run dev
```
