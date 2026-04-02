# 🎯 Quick Setup - New Users Start Here!

## 📦 **Just Cloned This Repo? Follow These Steps:**

### **1. Create `.env` file in project root**
```bash
copy .env.example .env
notepad .env
```

### **2. Add your FREE API keys to `.env`:**

```env
HF_TOKEN=hf_your_token_here                    # Get from https://huggingface.co/settings/tokens
GOOGLE_API_KEY=AIza_your_key_here             # Get from https://aistudio.google.com/apikey
```

### **3. Run with Docker:**
```bash
docker-compose up -d
```

### **4. Access:**
```
http://localhost:8000
```

---

## 📚 **Complete Guides:**

- **🏠 Local Setup (Docker):** [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) ← **START HERE**
- **🐳 Docker Troubleshooting:** [DOCKER_GUIDE.md](DOCKER_GUIDE.md)
- **🚀 Deploy to Cloud:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **⚡ Quick Docker Commands:** [DOCKER_QUICK_REFERENCE.md](DOCKER_QUICK_REFERENCE.md)

---

## 🔑 **Get Free API Keys:**

1. **HuggingFace (HF_TOKEN)** - Image generation
   - https://huggingface.co/settings/tokens
   - Free, no credit card

2. **Google Gemini (GOOGLE_API_KEY)** - Text generation
   - https://aistudio.google.com/apikey
   - Free, generous limits

3. **Groq (GROQ_API_KEY)** - Fast text generation (optional)
   - https://console.groq.com
   - Free tier available

---

## ⚡ **TL;DR - One Command Setup:**

```bash
# After creating .env with your API keys:
docker-compose up -d && docker-compose logs -f
```

Wait 5 minutes on first run (downloads AI models), then open http://localhost:8000

---

## 🆘 **Common Issues:**

| Problem | Solution |
|---------|----------|
| "HF_TOKEN: NOT SET" | Add `HF_TOKEN=hf_xxx` to `.env` file |
| Poor image quality | Missing HF_TOKEN in `.env` |
| Container crashes | Check `docker-compose logs` |
| Port 8000 in use | Stop other apps or change port in `docker-compose.yml` |

---

**See [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) for detailed step-by-step instructions!** 📖
