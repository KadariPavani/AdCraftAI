# Cloud Storage Migration - Setup Instructions

## ✅ FAISS Error Fixed
The `'key < ntotal' failed` error has been fixed in `app/pipeline.py`. 
The code now validates brand indices before reconstruction.

---

## 🚀 Cloud Migration Implementation

### Step 1: Install Dependencies

```bash
cd P:\KHUB\madverse
pip install cloudinary psycopg2-binary
```

### Step 2: Create Storage Module Directory

```bash
# Windows Command Prompt
mkdir app\storage

# Or PowerShell
New-Item -ItemType Directory -Path "app\storage" -Force
```

### Step 3: Create Storage Module Files

Create these 3 files in `app/storage/`:

#### File 1: `app/storage/__init__.py`
```python
"""
Storage module for cloud-based file management.
"""
from .cloudinary_manager import CloudinaryManager, get_cloudinary_manager
from .image_storage import ImageStorage

__all__ = ['CloudinaryManager', 'get_cloudinary_manager', 'ImageStorage']
```

#### File 2: `app/storage/cloudinary_manager.py`
See full code in: `CLOUD_SETUP_cloudinary_manager.py` (created separately)

#### File 3: `app/storage/image_storage.py`
See full code in: `CLOUD_SETUP_image_storage.py` (created separately)

### Step 4: Setup Cloudinary Account

1. **Sign up:** https://cloudinary.com/users/register/free
2. **Get credentials** from Dashboard:
   - Cloud Name
   - API Key  
   - API Secret

### Step 5: Update `.env` File

Add to your `.env` file:

```env
# ===================================
# Cloud Storage Configuration
# ===================================

# Cloudinary (Free: 25GB storage, 25GB bandwidth/month)
CLOUDINARY_CLOUD_NAME=your-cloud-name-here
CLOUDINARY_API_KEY=your-api-key-here
CLOUDINARY_API_SECRET=your-api-secret-here

# Enable cloud storage (true/false)
USE_CLOUD_STORAGE=true

# ===================================
# Database Configuration  
# ===================================

# Option 1: Keep SQLite for now (simplest)
# (No changes needed)

# Option 2: Migrate to Supabase PostgreSQL (recommended)
# DATABASE_URL=postgresql://user:password@host:5432/database
```

### Step 6: Update `app/pipeline.py`

Add at the top of the file (after other imports):

```python
from app.storage import ImageStorage
```

Modify the image saving logic in the `generate()` method:

```python
# OLD CODE (around line 350-360)
pamphlet_path = OUTPUT_DIR / f"{product_id}_{lang}_{timestamp}.png"
pamphlet.save(pamphlet_path)

# NEW CODE  
storage = ImageStorage()
pamphlet_url = storage.save_image(
    pamphlet,
    filename=f"{product_id}_{lang}_{timestamp}.png",
    folder="outputs"
)
# Now pamphlet_url is either a Cloudinary URL or local path
```

### Step 7: Update `app/database.py`

Change schema to store URLs instead of paths:

```python
# In create tables (line 38-47)
conn.execute("""
    CREATE TABLE IF NOT EXISTS generated_content (
        id TEXT PRIMARY KEY,
        product_id TEXT,
        content_type TEXT,
        language TEXT,
        content_json TEXT,
        pamphlet_url TEXT,  -- Changed from pamphlet_path
        product_image_url TEXT,  -- Changed from product_image_path
        created_at TEXT NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(id)
    )
""")
```

### Step 8: Test the Integration

```bash
# Start the server
python run.py

# Test with API request
POST http://localhost:8000/api/generate
{
  "prompt": "Nike running shoes",
  "languages": ["en"]
}
```

**Expected behavior:**
- If Cloudinary is configured → Images upload to cloud
- If not configured → Falls back to local storage
- Check logs for `[CLOUDINARY]` and `[STORAGE]` messages

### Step 9: Update Docker Configuration

Update `docker-compose.yml`:

```yaml
services:
  madverse:
    environment:
      # Add Cloudinary credentials
      - CLOUDINARY_CLOUD_NAME=${CLOUDINARY_CLOUD_NAME}
      - CLOUDINARY_API_KEY=${CLOUDINARY_API_KEY}
      - CLOUDINARY_API_SECRET=${CLOUDINARY_API_SECRET}
      - USE_CLOUD_STORAGE=true
    
    volumes:
      # Remove these (now in cloud)
      # - ./outputs:/app/outputs
      # - ./products_db:/app/products_db
      # - ./uploads:/app/uploads
      
      # Keep only cache volumes
      - model-cache:/root/.cache/huggingface

volumes:
  model-cache:
```

### Step 10: Rebuild and Test Docker

```bash
# Stop existing containers
docker-compose down -v

# Rebuild with new dependencies
docker-compose build --no-cache

# Start fresh
docker-compose up

# Test ad generation
# Restart container
# Verify data persists
```

---

## 📊 Migration Checklist

- [ ] Install cloudinary package
- [ ] Create app/storage directory
- [ ] Create 3 storage module files
- [ ] Sign up for Cloudinary
- [ ] Add credentials to .env
- [ ] Update pipeline.py (use ImageStorage)
- [ ] Update database.py (URLs not paths)
- [ ] Test locally
- [ ] Update docker-compose.yml
- [ ] Test in Docker
- [ ] Verify persistence after restart

---

## 🎯 Benefits After Migration

✅ **Generated ads persist** across Docker restarts
✅ **Fast delivery** via Cloudinary CDN
✅ **Shareable** - multiple instances use same storage
✅ **Free tier** - 25GB storage + bandwidth
✅ **Automatic backups** - Cloudinary handles it
✅ **No file system dependencies** - true cloud-native

---

## 🔧 Troubleshooting

**Images not uploading to Cloudinary?**
- Check `.env` has correct credentials
- Look for `[CLOUDINARY] ✅ Configured` in logs
- Check `USE_CLOUD_STORAGE=true` in `.env`

**Still using local storage?**
- Verify cloudinary package installed: `pip show cloudinary`
- Check logs for `[STORAGE] Using Cloudinary cloud storage`
- If see `[STORAGE] Using local file storage`, check credentials

**Docker not finding images?**
- Ensure environment variables passed to container
- Check `docker-compose.yml` has CLOUDINARY_* variables
- Verify `.env` file exists and is loaded

---

## 📚 Next Steps

**After basic migration works:**

1. **Database Migration**
   - Sign up for Supabase
   - Migrate to PostgreSQL
   - Update DATABASE_URL in .env

2. **FAISS Optimization**
   - Update BuildFAISS.py to store Cloudinary URLs
   - Rebuild index with cloud URLs
   - Test retrieval with remote images

3. **Production Deployment**
   - Deploy to Hugging Face Spaces
   - Or Railway/Render with Docker
   - All data persists in cloud

---

**Ready to start?** Run the commands above step by step!
