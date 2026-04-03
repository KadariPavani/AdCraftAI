# 🚨 QUICK FIX: FAISS Index Error

## Error Message:
```
Generation errors: Pipeline error: Error in virtual void faiss::IndexFlat::reconstruct
faiss::idx_t, float*) const at /project/third-party/faiss/faiss/IndexFlat.cpp:299:
Error: 'key < ntotal' failed
```

## Root Cause:
**Git LFS files not downloaded!** You have Git LFS pointer files instead of actual data.

---

## ✅ Solution (5 Steps):

### 1. Install Git LFS
```bash
# Windows: Download from https://git-lfs.github.com/
# Linux:
sudo apt-get install git-lfs
# Mac:
brew install git-lfs

# Then run:
git lfs install
```

### 2. Pull Large Files
```bash
cd /path/to/MAdVerse
git lfs pull
```

### 3. Verify Files Downloaded
```bash
python scripts/check_faiss.py
```

**Expected output:**
```
✓ ALL CHECKS PASSED!
```

**If you still see errors**, the files are still pointers. Try:
```bash
git lfs fetch --all
git lfs checkout
```

### 4. Rebuild Docker (IMPORTANT!)
```bash
docker-compose down
docker-compose up -d --build
```

### 5. Verify Working
```bash
# Check logs
docker-compose logs -f

# Test health
curl http://localhost:8000/api/health

# Should return: {"status":"healthy"}
```

---

## How to Prevent This:

**Always clone with Git LFS installed:**
```bash
git lfs install
git clone https://github.com/YOUR_USERNAME/MAdVerse.git
cd MAdVerse
python scripts/check_faiss.py
```

---

## File Size Reference:

If files are correct, you should see:

| File | Expected Size |
|------|---------------|
| `embeddings/faiss_indexes/madverse_index.faiss` | ~250+ MB |
| `embeddings/faiss_indexes/id_to_metadata.pkl` | ~10+ MB |
| `processed/metadata/madverse_metadata.csv` | ~10+ MB |

**If any file is < 1 KB, it's a Git LFS pointer!**

To check:
```bash
ls -lh embeddings/faiss_indexes/
ls -lh processed/metadata/
```

---

## Still Not Working?

See full troubleshooting guide: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

Or run diagnostics:
```bash
# Full validation
python scripts/check_faiss.py

# Check what's in the files
head -n 1 embeddings/faiss_indexes/madverse_index.faiss

# If it shows "version https://git-lfs.github.com", it's a pointer!
```

---

**TL;DR:** `git lfs install → git lfs pull → docker-compose up -d --build`
