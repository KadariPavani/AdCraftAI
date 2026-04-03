# Cloud Storage Migration Plan for MAdVerse

## Problem Statement

**Current Issues:**
1. ❌ Local file storage (folders: `outputs/`, `uploads/`, `data/images/`, `products_db/`)
2. ❌ Docker containers lose generated products on restart
3. ❌ Dataset not shareable across deployments
4. ❌ FAISS index references local file paths
5. ❌ SQLite database stored locally, not accessible across instances
6. ❌ Not scalable for production/cloud deployment

**Goal:**
✅ Move to cloud-based, persistent, shareable storage
✅ Products persist across Docker restarts
✅ Multiple instances can share the same dataset
✅ Production-ready architecture

---

## Architecture Design

### Current Architecture (Local)
```
┌─────────────────────────────────────────────┐
│ Docker Container                             │
│                                              │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │   FastAPI    │──────│  Local Folders  │ │
│  │   Pipeline   │      │  - outputs/     │ │
│  └──────────────┘      │  - uploads/     │ │
│         │              │  - data/images/ │ │
│         │              │  - products_db/ │ │
│         │              └─────────────────┘ │
│  ┌──────▼──────┐                           │
│  │ SQLite DB   │ (lost on restart)         │
│  └─────────────┘                           │
└─────────────────────────────────────────────┘
```

### New Architecture (Cloud)
```
┌─────────────────────────────────────────────┐
│ Docker Container (Stateless)                 │
│                                              │
│  ┌──────────────┐                           │
│  │   FastAPI    │                           │
│  │   Pipeline   │                           │
│  └───┬──────┬───┘                           │
│      │      │                                │
└──────┼──────┼────────────────────────────────┘
       │      │
       │      └──────────────────┐
       │                         │
   ┌───▼────────────┐    ┌──────▼─────────────┐
   │  Cloudinary    │    │  Cloud Database    │
   │  (Images)      │    │  (PostgreSQL)      │
   │                │    │                    │
   │  - Outputs     │    │  - Products        │
   │  - Uploads     │    │  - Generated Ads   │
   │  - Dataset     │    │  - Metadata        │
   │  - Logos       │    │  - Analytics       │
   └────────────────┘    └────────────────────┘
         ↓                       ↓
   ┌─────────────────────────────────────┐
   │  FAISS Index (Cloud URLs)           │
   │  - Metadata stores Cloudinary URLs  │
   │  - Images fetched on-demand         │
   └─────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Cloud Storage Analysis ✅
**Status:** Ready to start

**Tasks:**
- [x] Document current file storage usage
- [ ] Estimate storage requirements
- [ ] Calculate costs (Cloudinary free tier: 25GB storage, 25GB bandwidth)
- [ ] Identify which files need cloud migration vs. can stay in Docker volumes

**Files to migrate:**
1. **Critical (must be cloud):**
   - `outputs/` - Generated ads (need persistence)
   - `data/images/` - Dataset images (50K+ images)
   - Generated product images

2. **Should be cloud:**
   - `uploads/` - User uploads (temporary, can use cloud or temp volume)

3. **Can stay local/volume:**
   - `embeddings/` - Large binary files, rebuild on deployment
   - FAISS index - Can be in Docker volume, rebuild if needed

---

### Phase 2: Design Cloud Architecture ⏳
**Status:** Pending analysis

**Decisions needed:**
1. **Image Storage:** Cloudinary (recommended) vs. AWS S3 vs. Azure Blob
2. **Database:** 
   - Supabase (PostgreSQL + free tier)
   - Railway (PostgreSQL)
   - Render (PostgreSQL)
   - PlanetScale (MySQL)
3. **FAISS Strategy:**
   - Option A: Store in Docker volume, rebuild on deploy
   - Option B: Store on S3/cloud storage, download on startup
   - Option C: Use vector database (Pinecone, Qdrant, Weaviate)

**Recommended Stack:**
- 🖼️ **Images:** Cloudinary (free tier, CDN, transformations)
- 🗄️ **Database:** Supabase (free PostgreSQL, 500MB, good for MVP)
- 📊 **FAISS:** Docker volume + rebuild script (fastest, simplest)
- 🔄 **Alternative:** Qdrant Cloud (vector DB, replaces FAISS entirely)

---

### Phase 3: Implement Cloudinary Integration ⏳
**Status:** Pending design

**Files to create/modify:**
1. `app/storage/cloudinary_manager.py` - Upload/download helper
2. `app/storage/image_storage.py` - Abstract storage interface
3. Update `app/pipeline.py` - Use cloud storage for outputs
4. Update `app/database.py` - Store Cloudinary URLs, not paths

**Code structure:**
```python
class CloudinaryManager:
    def upload_image(self, image_path, folder="outputs") -> str:
        """Upload image, return Cloudinary URL"""
    
    def download_image(self, url, local_path=None) -> Image:
        """Download from URL, return PIL Image"""
    
    def delete_image(self, public_id):
        """Delete from Cloudinary"""
```

**Environment variables:**
```env
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
```

---

### Phase 4: Migrate to Cloud Database ⏳
**Status:** Pending design

**Database schema migration:**
```sql
-- Products table (store Cloudinary URLs)
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2),
    category VARCHAR(100),
    brand VARCHAR(100),
    image_urls JSONB,  -- Array of Cloudinary URLs
    enhanced_image_url TEXT,  -- Cloudinary URL
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Generated content (store Cloudinary URLs)
CREATE TABLE generated_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID REFERENCES products(id),
    content_type VARCHAR(50),
    language VARCHAR(10),
    content_json JSONB,
    pamphlet_url TEXT,  -- Cloudinary URL
    product_image_url TEXT,  -- Cloudinary URL
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Files to modify:**
1. `app/database.py` - Replace SQLite with PostgreSQL
2. `requirements.txt` - Add `psycopg2-binary` or `asyncpg`
3. `.env` - Add database connection string

---

### Phase 5: Update FAISS for Cloud URLs ⏳
**Status:** Pending Cloudinary implementation

**Changes:**
1. Modify `BuildFAISS.py` - Store Cloudinary URLs in metadata
2. Modify `app/pipeline.py` - Fetch images from URLs when needed
3. Update `id_to_metadata` structure:
   ```python
   {
       'image_id': 'abc123',
       'image_url': 'https://res.cloudinary.com/...jpg',  # Changed from path
       'source': 'dataset',
       'category': 'jewelry',
       'brand': 'Kalyan_Jewellers'
   }
   ```

**FAISS deployment strategy:**
```dockerfile
# In Dockerfile
RUN python BuildFAISS.py  # Build index during image build
# OR
CMD ["sh", "-c", "python BuildFAISS.py && python run.py"]  # Build on startup
```

---

### Phase 6: Fix Docker Persistence ⏳
**Status:** Pending database migration

**Update `docker-compose.yml`:**
```yaml
services:
  madverse:
    environment:
      # Cloudinary
      - CLOUDINARY_CLOUD_NAME=${CLOUDINARY_CLOUD_NAME}
      - CLOUDINARY_API_KEY=${CLOUDINARY_API_KEY}
      - CLOUDINARY_API_SECRET=${CLOUDINARY_API_SECRET}
      
      # Database (Supabase/Railway/Render)
      - DATABASE_URL=${DATABASE_URL}
      
      # Storage strategy
      - USE_CLOUD_STORAGE=true
    
    volumes:
      # Only keep cache/temp volumes
      - faiss-index:/app/embeddings/faiss_indexes
      - model-cache:/root/.cache/huggingface
    
    restart: unless-stopped

volumes:
  faiss-index:
  model-cache:
```

**Remove local volume mounts:**
- ❌ `./outputs:/app/outputs` (now in Cloudinary)
- ❌ `./products_db:/app/products_db` (now in PostgreSQL)
- ❌ `./uploads:/app/uploads` (now in Cloudinary)

---

### Phase 7: Testing & Validation ⏳
**Status:** Pending all implementations

**Test scenarios:**
1. ✅ Generate ad → verify stored in Cloudinary
2. ✅ Generate ad → verify metadata in PostgreSQL
3. ✅ Restart Docker → verify data persists
4. ✅ Deploy new container → verify can access previous data
5. ✅ Generate ad with brand search → verify FAISS works with URLs
6. ✅ Load test → verify Cloudinary CDN performance

---

## Cost Estimation

### Free Tier Limits
**Cloudinary Free:**
- 25 GB storage
- 25 GB/month bandwidth
- 25k transformations/month
- ✅ Sufficient for 50k images (~500MB) + generated outputs

**Supabase Free:**
- 500 MB database
- Unlimited API requests
- 2 GB bandwidth/month
- ✅ Sufficient for metadata + generated content

**Total Cost:** $0/month (within free tiers)

**When to upgrade:**
- Cloudinary: If dataset + outputs > 25GB
- Supabase: If metadata > 500MB or traffic > 2GB/month

---

## Migration Steps (Execution Order)

### Step 1: Setup Cloud Accounts
```bash
1. Sign up for Cloudinary (https://cloudinary.com)
2. Sign up for Supabase (https://supabase.com)
3. Get API credentials for both
4. Add to .env file
```

### Step 2: Implement Cloudinary
```bash
pip install cloudinary
python scripts/migrate_images_to_cloudinary.py  # Upload existing dataset
```

### Step 3: Migrate Database
```bash
pip install psycopg2-binary
python scripts/migrate_db_to_postgres.py  # Migrate SQLite → PostgreSQL
```

### Step 4: Update FAISS
```bash
python BuildFAISS.py  # Rebuild with Cloudinary URLs
```

### Step 5: Test
```bash
docker-compose down -v  # Remove all volumes
docker-compose up --build  # Fresh start
# Test ad generation
# Restart container
# Verify data persists
```

---

## Benefits After Migration

✅ **Persistence:** Products survive Docker restarts
✅ **Scalability:** Can run multiple containers sharing same data
✅ **Performance:** Cloudinary CDN for fast image delivery
✅ **Backups:** Cloud providers handle backups automatically
✅ **Collaboration:** Team can access same dataset
✅ **Production-ready:** No local file dependencies
✅ **Cost-effective:** Free tiers cover MVP needs

---

## Next Steps

**Immediate (This session):**
1. ✅ Fix FAISS error (DONE)
2. ⏳ Set up Cloudinary account
3. ⏳ Implement basic Cloudinary integration
4. ⏳ Test image upload/download

**Short-term (Next session):**
1. Set up Supabase database
2. Migrate database schema
3. Update FAISS metadata structure
4. Test end-to-end

**Long-term:**
1. Optimize cloud costs
2. Implement caching layer
3. Add CDN for FAISS index
4. Consider vector database migration

---

## Files to Create

1. `app/storage/cloudinary_manager.py` - Cloudinary operations
2. `app/storage/image_storage.py` - Storage abstraction
3. `app/storage/__init__.py` - Storage module
4. `scripts/migrate_images_to_cloudinary.py` - One-time migration
5. `scripts/migrate_db_to_postgres.py` - Database migration
6. `scripts/rebuild_faiss_cloud.py` - Rebuild FAISS with URLs
7. `.env.example` - Updated with cloud credentials
8. `CLOUD_DEPLOYMENT.md` - Cloud deployment docs

---

**Ready to start implementation?** Let me know if you want to:
1. Start with Cloudinary integration
2. Set up database first
3. Review the architecture more
4. Adjust the plan
