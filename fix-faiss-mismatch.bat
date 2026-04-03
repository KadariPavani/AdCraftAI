@echo off
REM ============================================================
REM Fix FAISS Index - Copy from Original Project
REM ============================================================

echo ============================================================
echo   FIXING FAISS INDEX MISMATCH
echo ============================================================
echo.

echo [Step 1] Backing up current FAISS files...
if exist embeddings\faiss_indexes\madverse_index.faiss.backup del embeddings\faiss_indexes\madverse_index.faiss.backup
if exist embeddings\faiss_indexes\id_to_metadata.pkl.backup del embeddings\faiss_indexes\id_to_metadata.pkl.backup

copy embeddings\faiss_indexes\madverse_index.faiss embeddings\faiss_indexes\madverse_index.faiss.backup
copy embeddings\faiss_indexes\id_to_metadata.pkl embeddings\faiss_indexes\id_to_metadata.pkl.backup
echo   - Backup created

echo.
echo [Step 2] Copying fresh FAISS files from original project...
copy /Y P:\KHUB\product-labs\MAdVerse\embeddings\faiss_indexes\madverse_index.faiss embeddings\faiss_indexes\madverse_index.faiss
copy /Y P:\KHUB\product-labs\MAdVerse\embeddings\faiss_indexes\id_to_metadata.pkl embeddings\faiss_indexes\id_to_metadata.pkl
copy /Y P:\KHUB\product-labs\MAdVerse\embeddings\image_embeddings.pkl embeddings\image_embeddings.pkl
echo   - Fresh FAISS files copied

echo.
echo [Step 3] Verifying FAISS files...
python P:\KHUB\product-labs\MAdVerse\scripts\check_faiss.py

echo.
echo [Step 4] Committing fixed files...
git add embeddings/
git commit -m "Fix: Update FAISS index and metadata files"

echo.
echo [Step 5] Pushing to Hugging Face...
git push origin main

echo.
echo ============================================================
echo   DONE! Space will restart with fixed FAISS files.
echo ============================================================
echo.
echo Wait 2-3 minutes for Space to restart, then test again.
pause
