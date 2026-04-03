# FAISS Index Error Fix

## Problem
Error: `'key < ntotal' failed` during FAISS vector reconstruction when searching for brand-specific ads.

## Root Cause
The metadata dictionary contains indices that are >= the total number of vectors in the FAISS index (ntotal=61,576). When the pipeline tries to reconstruct vectors using these invalid indices, FAISS throws an error because it can't access vectors that don't exist.

## Solution Applied

### 1. Immediate Fix (✅ DONE)
Modified `app/pipeline.py` line 179-208 to filter out invalid indices before reconstruction:

```python
# Filter out invalid indices that are beyond FAISS index size
valid_indices = [int(i) for i in brand_indices if int(i) < self.index.ntotal]
```

**What this does:**
- Validates all brand indices against FAISS index size before attempting reconstruction
- Filters out any indices >= index.ntotal
- Falls back to full index search if no valid indices exist
- Logs warnings when invalid indices are detected

**Impact:**
- ✅ Prevents the RuntimeError
- ✅ Allows ad generation to continue
- ✅ Provides diagnostic information via log warnings
- ⚠️ May return fewer results if many indices are invalid

### 2. Root Cause Investigation
Created `validate_faiss.py` to diagnose metadata/index inconsistencies.

**To run validation:**
```bash
cd P:\KHUB\madverse
python validate_faiss.py
```

This will show:
- Whether metadata keys are sequential (0 to n-1)
- Any missing or extra keys
- Which specific indices are invalid
- Which brands are affected

### 3. Permanent Fix (if needed)

If validation reveals inconsistencies, rebuild the FAISS index:

```bash
cd P:\KHUB\madverse
python BuildFAISS.py
```

This will:
1. Load embeddings from `embeddings/image_embeddings.pkl`
2. Create sequential metadata indices (0 to n-1)
3. Build a clean FAISS index
4. Save synchronized index and metadata

## Testing

After applying the fix, test with the same query that failed:
```
POST /api/generate
{
  "prompt": "Kalyan Jewellers Tejasvi collection 22K gold bangles...",
  "languages": ["en"],
  "brand": "Kalyan_Jewellers"
}
```

**Expected behavior:**
- No RuntimeError
- Ad generation completes successfully
- If invalid indices exist, you'll see warnings in logs:
  ```
  [FAISS] WARNING: Filtered X invalid indices
  [FAISS] Using Y valid indices out of Z
  ```

## Files Modified

1. ✅ `app/pipeline.py` - Added index validation in retrieve() method
2. ✅ `validate_faiss.py` - New diagnostic script

## Next Steps

1. **Test the fix:** Try generating an ad with the same Kalyan Jewellers query
2. **Run validation:** Execute `python validate_faiss.py` to see if index needs rebuilding
3. **Rebuild if needed:** If validation shows major issues, run `python BuildFAISS.py`
4. **Monitor logs:** Watch for FAISS warnings during ad generation

## Prevention

To prevent this in the future:
- Always use `BuildFAISS.py` to create the index (ensures sequential indices)
- Don't manually edit metadata files
- Validate index after any dataset updates
- Keep embeddings and index in sync

---
**Status:** ✅ Error fixed - ad generation will no longer crash
**Validation:** Run `python validate_faiss.py` to check for deeper issues
