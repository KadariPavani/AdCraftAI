# 🐳 Docker Setup Guide for MAdVerse
## Fix Image Generation & Database Access in Docker

---

## 🔧 **The Problem**

When running in Docker:
- ❌ `.env` file is blocked by `.dockerignore` 
- ❌ HF_TOKEN and API keys not available
- ❌ Image generation produces poor quality (fallback to gradient)
- ❌ Database doesn't persist between restarts

---

## ✅ **The Solution**

I've created **3 ways** to run Docker with proper environment variables:

---

## 🚀 **Option 1: Docker Compose (RECOMMENDED)**

### Step 1: Make sure your `.env` file exists
```bash
# Check if .env exists
dir .env

# If not, copy from example
copy .env.example .env

# Edit .env and add your API keys
notepad .env
```

Add your keys to `.env`:
```env
# Required for best image generation
HF_TOKEN=hf_your_token_here
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_key

# Optional
TOGETHER_API_KEY=your_together_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Step 2: Run with Docker Compose
```bash
# Start the container (builds automatically)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**✅ Benefits:**
- Automatically loads `.env` file
- Persists database, outputs, and uploads
- Easy to start/stop
- Proper port mapping (8000)

---

## 🚀 **Option 2: Simple Docker Run (Windows)**

Just double-click: `docker-run.bat`

Or run manually:
```bash
docker-run.bat
```

This script:
- Builds the Docker image
- Loads your `.env` file automatically
- Mounts volumes for persistence
- Starts on port 8000

---

## 🚀 **Option 3: Manual Docker Run**

If you want full control:

```bash
# Build
docker build -t madverse-app .

# Run with .env file
docker run -d ^
  --name madverse-app ^
  -p 8000:8000 ^
  --env-file .env ^
  -e PORT=8000 ^
  -v "%cd%\products_db:/app/products_db" ^
  -v "%cd%\outputs:/app/outputs" ^
  -v "%cd%\uploads:/app/uploads" ^
  madverse-app

# View logs
docker logs -f madverse-app

# Stop
docker stop madverse-app

# Remove
docker rm madverse-app
```

---

## 📂 **Volume Mounts Explained**

These volumes ensure data persists:

```yaml
-v ./products_db:/app/products_db    # SQLite database
-v ./outputs:/app/outputs            # Generated ads
-v ./uploads:/app/uploads            # User-uploaded images
```

**Without volumes:**
- Database resets on container restart ❌
- Generated images lost ❌
- Uploads lost ❌

**With volumes:**
- Everything persists ✅
- Data accessible on host ✅

---

## 🔑 **Environment Variables**

### Method 1: `.env` file (Recommended)
```env
HF_TOKEN=hf_xxxxx
GOOGLE_API_KEY=AIzaxxxxx
PORT=8000
```

Load with:
```bash
docker-compose up
# OR
docker run --env-file .env ...
```

### Method 2: Individual `-e` flags
```bash
docker run \
  -e HF_TOKEN=hf_xxxxx \
  -e GOOGLE_API_KEY=AIzaxxxxx \
  -e PORT=8000 \
  madverse-app
```

### Method 3: Pass at runtime (not recommended - insecure)
```bash
docker run \
  -e HF_TOKEN=$(cat .env | grep HF_TOKEN | cut -d '=' -f2) \
  madverse-app
```

---

## 🎨 **Image Generation Quality**

| Scenario | Quality | Speed |
|----------|---------|-------|
| **With HF_TOKEN** | ⭐⭐⭐⭐⭐ (FLUX.1-schnell) | 5-10s |
| **With TOGETHER_API_KEY** | ⭐⭐⭐⭐ (FLUX via Together) | 3-5s |
| **No API keys** | ⭐⭐ (Gradient fallback - solid colors) | Instant |

---

## 🗄️ **Database Access**

The SQLite database is mounted at `./products_db/adcraft.db`.

**Access from host:**
```bash
# Install SQLite browser
# Open: products_db/adcraft.db
```

**Access from container:**
```bash
docker exec -it madverse-app sqlite3 /app/products_db/adcraft.db
```

---

## 🔍 **Verify Everything Works**

### 1. Check container is running
```bash
docker ps
```

### 2. Check logs
```bash
docker logs madverse-app
```

Look for:
```
[INIT] HF_TOKEN: set (hf_xxxxx...)  ✅
[INIT] TOGETHER_API_KEY: set ✅
[INIT] ImageGenerator initialized: Primary: HuggingFace FLUX.1-schnell (token available) ✅
```

### 3. Test health endpoint
```bash
curl http://localhost:8000/api/health
```

Should return:
```json
{"status": "healthy", "version": "2.1.0"}
```

### 4. Test image generation
```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Red Bull energy drink\", \"num_variations\": 1}"
```

Check `outputs/` folder for generated image.

---

## 🐛 **Troubleshooting**

### Problem: "HF_TOKEN: NOT SET"
**Solution:** 
```bash
# Make sure .env exists
cat .env | grep HF_TOKEN

# Restart with --env-file
docker-compose down && docker-compose up -d
```

### Problem: Database resets on restart
**Solution:** 
```bash
# Check volumes are mounted
docker inspect madverse-app | grep -A 10 Mounts

# Should see: products_db mounted
```

### Problem: Generated images not visible
**Solution:**
```bash
# Check outputs volume
dir outputs

# Check container can write
docker exec madverse-app ls -la /app/outputs
```

### Problem: Poor image quality
**Solution:**
```bash
# Add HF_TOKEN to .env
echo HF_TOKEN=hf_your_token_here >> .env

# Restart
docker-compose restart
```

---

## 🎯 **Quick Start Commands**

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Restart (e.g., after changing .env)
docker-compose restart

# Stop
docker-compose down

# Stop and remove volumes (DANGER: deletes all data)
docker-compose down -v
```

---

## 🌐 **Access URLs**

Once running:
- **App:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/api/health
- **Dataset Explorer:** http://localhost:8000/dataset

---

## 📊 **Resource Usage**

Typical container usage:
- **RAM:** 2-4 GB (with models loaded)
- **CPU:** 1-2 cores
- **Disk:** 2-3 GB (app + models)

**Recommended host specs:**
- 8 GB RAM minimum
- 10 GB free disk
- Docker Desktop 4.0+

---

## 🔒 **Security Notes**

1. **Never commit `.env`** - it's in `.gitignore` ✅
2. **Never share your API keys**
3. **`.dockerignore` blocks `.env`** - use `--env-file` instead ✅
4. Use environment variables, not hardcoded keys

---

## 📦 **Files Created**

- ✅ `docker-compose.yml` - Compose configuration
- ✅ `docker-run.bat` - Windows run script  
- ✅ `docker-run.sh` - Linux/Mac run script
- ✅ `DOCKER_GUIDE.md` - This guide

---

## ✨ **Summary**

**To run with full features:**

```bash
# 1. Create .env with your API keys
copy .env.example .env
notepad .env

# 2. Start with Docker Compose
docker-compose up -d

# 3. Access
# http://localhost:8000
```

**That's it!** 🎉

Your Docker container now has:
- ✅ All API keys loaded
- ✅ High-quality image generation
- ✅ Persistent database
- ✅ Persistent outputs & uploads
