# 🔧 FAISS Index Mismatch - FIXED!

## ❌ The Problem

You're getting this error on HuggingFace Space:
```
Pipeline error: Error in virtual void faiss::IndexFlat::reconstruct(faiss::idx_t, float*) const 
at /project/third-party/faiss/faiss/IndexFlat.cpp:299: Error: 'key < ntotal' failed
```

**Root cause:**
```
FAISS Index: 61,628 vectors
Metadata: 61,630 entries (2 EXTRA!)
```

**Mismatch:** Trying to access vectors 61,628 and 61,629 that don't exist in the index!

---

## Solution: Copy Fresh FAISS Files

### Quick Fix Script

```powershell
.\fix-faiss-mismatch.bat
```

---

### Manual Fix

**Step 1: Navigate to madverse folder**
```powershell
cd P:\KHUB\madverse
```

**Step 2: Verify original files are correct**
```powershell
cd P:\KHUB\product-labs\MAdVerse
python scripts\check_faiss.py
```

You should see:
```
✓ FAISS index OK (345 MB)
✓ Metadata file OK
✓ Vectors and metadata match
```

**Step 3: Copy fresh files to madverse**
```powershell
cd P:\KHUB\madverse

# Copy FAISS index
Copy-Item -Force P:\KHUB\product-labs\MAdVerse\embeddings\faiss_indexes\madverse_index.faiss embeddings\faiss_indexes\madverse_index.faiss

# Copy metadata
Copy-Item -Force P:\KHUB\product-labs\MAdVerse\embeddings\faiss_indexes\id_to_metadata.pkl embeddings\faiss_indexes\id_to_metadata.pkl

# Copy embeddings
Copy-Item -Force P:\KHUB\product-labs\MAdVerse\embeddings\image_embeddings.pkl embeddings\image_embeddings.pkl
```

**Step 4: Verify files are correct**
```powershell
python scripts\check_faiss.py
```

**Step 5: Push to Hugging Face**
```powershell
git add embeddings/
git commit -m "Fix: Update FAISS index with correct files"
git push origin main
```

**Step 6: Wait for restart (2-3 minutes)**

---

## Why This Happens

**Possible causes:**
1. FAISS files got corrupted during copy
2. Partial Git LFS download (didn't get full files)
3. Wrong version of files copied

**The fix:**
- Use fresh files from original working project
- Ensure Git LFS uploaded them correctly
- Verify vectors and metadata match

---

## Verify After Fix

**Check logs for:**
```
[FAISS] Index loaded: 61,628 vectors
[FAISS] Metadata entries: 61,628  ← Should MATCH!
```

**Test ad generation:**
- Try generating an ad
- Should work without FAISS errors
- If still errors, check logs for details

---

## About Dataset Re-downloading

If dataset downloads again on restart, it means the `.download_complete` marker was lost.

**To prevent dataset download:**

Add this to your Space's environment variables:
```
SKIP_DATASET_DOWNLOAD=1
```

This is already in the Dockerfile, but you can also set it in Space Settings → Variables:
- Name: `SKIP_DATASET_DOWNLOAD`
- Value: `1`

**Why this works:**
- Embeddings work WITHOUT the original dataset images
- The FAISS index already contains all the information
- Dataset images are only needed if you want to display original ads

---

## Summary

**Current issues:**
1. ✅ FAISS mismatch → Copy fresh files from original project
2. ✅ Dataset re-download → Set `SKIP_DATASET_DOWNLOAD=1`

**After fix:**
- FAISS vectors and metadata will match
- Ad generation will work
- No more download on restart

---

**Run the fix now!**
```powershell
.\fix-faiss-mismatch.bat
```
