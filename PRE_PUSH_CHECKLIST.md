# 🧹 Pre-Push Cleanup Checklist for MAdVerse

## ⚠️ **CRITICAL: Git LFS Validation**

Before pushing, verify Git LFS files are properly tracked:

```bash
# 1. Ensure Git LFS is installed
git lfs install

# 2. Validate FAISS files
python scripts/check_faiss.py

# 3. Check what's tracked by LFS
git lfs ls-files

# 4. Verify file sizes (should be large, not pointer files)
ls -lh embeddings/faiss_indexes/madverse_index.faiss
# Should show ~250+ MB, NOT < 1 KB
```

**Expected output:**
```
embeddings/faiss_indexes/madverse_index.faiss (250+ MB)
embeddings/faiss_indexes/id_to_metadata.pkl (10+ MB)
processed/metadata/madverse_metadata.csv (10+ MB)
```

---

## ✅ **Files SAFE to Push (Already in .gitignore):**

These will NOT be pushed (protected by `.gitignore`):
- ✅ `.env` - Your API keys (NEVER pushed)
- ✅ `.venv/` - Virtual environment
- ✅ `__pycache__/` - Python cache
- ✅ `outputs/` - Generated ads
- ✅ `products_db/` - SQLite database
- ✅ `uploads/` - User uploads
- ✅ `data/images/` - Large dataset images
- ✅ `.vscode/`, `.idea/` - IDE settings
- ✅ `*.log` - Log files

## 📦 **Files WILL be Pushed (Required for deployment):**

- ✅ `embeddings/` - Pre-computed embeddings (~410 MB)
- ✅ `data/annotations/` - Dataset metadata (~12 MB)
- ✅ `data/indices/` - Statistics files
- ✅ `processed/` - Processed metadata (~12 MB)
- ✅ All `.py` files - Application code
- ✅ `Dockerfile`, `docker-compose.yml` - Docker configs
- ✅ All documentation (`.md` files)
- ✅ `.env.example` - Template (no secrets)
- ✅ `requirements.txt` - Python dependencies

## 🗑️ **Optional: Files You Might Want to Remove**

These are in your repo but not needed:

```bash
# Word documents (keep or remove)
AdCraft.docx
Pavani_ProductLabs.docx
WEEKLY_DOCUMENTATION.docx

# Batch files (keep FIX_DOCKER.bat, remove others if not used)
docker-run.bat (optional - documented in guides)
```

## 🚀 **Commands to Clean and Push**

### **Step 1: Check Current Status**
```bash
git status
```

### **Step 2: Remove Unwanted Files (Optional)**

If you want to remove the .docx files:
```bash
git rm AdCraft.docx Pavani_ProductLabs.docx WEEKLY_DOCUMENTATION.docx
```

### **Step 3: Add All Changes**
```bash
git add .
```

### **Step 4: Verify What Will Be Committed**
```bash
git status
```

**Make sure you DON'T see:**
- ❌ `.env` (should be ignored)
- ❌ `outputs/` folder
- ❌ `products_db/` folder
- ❌ `.venv/` folder

**Should see:**
- ✅ New `.md` documentation files
- ✅ Updated `Dockerfile`
- ✅ Updated `docker-compose.yml`
- ✅ New scripts

### **Step 5: Commit**
```bash
git commit -m "Add Docker setup with environment variable support and comprehensive documentation"
```

### **Step 6: Push to GitHub**
```bash
git push origin main
```

---

## 🔒 **CRITICAL: Verify .env is NOT Being Pushed**

Before pushing, run:
```bash
git status | findstr ".env"
```

**Should NOT show `.env` file!**

If it does show `.env`:
```bash
git reset HEAD .env
git update-index --assume-unchanged .env
```

---

## 📊 **Expected Commit Size**

- **Documentation files:** ~30 KB
- **Docker configs:** ~10 KB
- **Code changes:** ~5 KB
- **Scripts:** ~5 KB
- **Total:** ~50 KB

**Note:** If you're pushing `embeddings/` for the first time, it will be ~410 MB (one-time).

---

## ⚠️ **Large File Warning**

If `embeddings/image_embeddings.pkl` is too large (>100 MB), GitHub might reject it.

**Solution: Use Git LFS**
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "embeddings/*.pkl"
git lfs track "embeddings/faiss_indexes/*"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add embeddings/
git commit -m "Add embeddings with Git LFS"
git push
```

---

## 🎯 **Quick Clean Push (Safe Commands)**

```bash
# 1. Check status
git status

# 2. Add new files only (safe)
git add SETUP.md LOCAL_SETUP_GUIDE.md DOCKER_GUIDE.md DOCKER_QUICK_REFERENCE.md DEPLOYMENT_GUIDE.md
git add docker-compose.yml Dockerfile FIX_DOCKER.bat docker-run.bat docker-run.sh
git add scripts/start.sh

# 3. Commit
git commit -m "feat: Add Docker support with comprehensive setup guides

- Add docker-compose.yml for easy local setup
- Update Dockerfile to fix line ending issues
- Add environment variable support via .env file
- Add comprehensive documentation:
  - LOCAL_SETUP_GUIDE.md for new users
  - DOCKER_GUIDE.md for troubleshooting
  - DOCKER_QUICK_REFERENCE.md for quick commands
  - DEPLOYMENT_GUIDE.md for cloud deployment
  - SETUP.md as quick start guide
- Add helper scripts for Docker setup
- Skip dataset download by default (use embeddings only)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

# 4. Push
git push origin main
```

---

## ✅ **Post-Push Verification**

After pushing, verify on GitHub:

1. ✅ `.env` is NOT visible
2. ✅ Documentation files are there
3. ✅ `docker-compose.yml` is updated
4. ✅ `Dockerfile` is updated
5. ✅ Clone in a new folder and test setup

---

## 🧪 **Test After Push**

```bash
# Clone to new location
cd /tmp
git clone https://github.com/YOUR_USERNAME/MAdVerse.git test-clone
cd test-clone

# Follow SETUP.md
copy .env.example .env
notepad .env  # Add your keys

# Test Docker
docker-compose up -d
docker-compose logs -f
```

---

## 📝 **Commit Message Template**

```
feat: Add Docker support with comprehensive setup guides

- Add docker-compose.yml for easy local setup
- Update Dockerfile to fix line ending issues  
- Add environment variable support via .env file
- Add comprehensive documentation
- Skip dataset download (use embeddings only)
- Fix image generation in Docker

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

---

## 🎉 **You're Ready to Push!**

Your repository now has:
- ✅ Working Docker setup
- ✅ Environment variable support
- ✅ Comprehensive documentation
- ✅ Proper .gitignore (protects secrets)
- ✅ Easy setup for new users

**Run the commands above and push!** 🚀
