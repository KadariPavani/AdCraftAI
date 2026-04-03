# Fix: Remove exposed secrets from git history and push clean version

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "FIXING EXPOSED SECRETS IN GIT HISTORY" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Removing last commit with secrets..." -ForegroundColor Yellow
git reset --soft HEAD~1
Write-Host "Done!" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Staging cleaned files..." -ForegroundColor Yellow
git add .
Write-Host "Done!" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Creating new commit (without secrets)..." -ForegroundColor Yellow
git commit -m "feat: Add Cloudinary storage + Fix FAISS sync (secrets removed)

Cloudinary Integration:
- Add ImageStorage abstraction for cloud/local storage
- Integrate storage in pipeline.py and main.py
- Replace all PIL save() calls with storage.save_image()
- Automatic fallback to local storage if Cloudinary fails

FAISS Index Fix:
- Remove 2 orphaned metadata entries (keys 61628-61629)
- Align metadata count with FAISS vector count (61628)
- Improve persistence with atomic writes
- Prevent future metadata/index mismatches

Documentation:
- Add deployment guides with placeholder values
- Update README_HF_SPACE.md with setup instructions
- Create .env.example for configuration reference

Security: All actual API keys removed from documentation

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
Write-Host "Done!" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4: Pushing to HuggingFace..." -ForegroundColor Yellow
git push origin main --force
Write-Host ""

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "COMPLETE!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Check if push succeeded" -ForegroundColor White
Write-Host "2. Add secrets manually in HF Space Settings (Repository secrets)" -ForegroundColor White
Write-Host "3. IMPORTANT: Regenerate HF_TOKEN for security!" -ForegroundColor Red
Write-Host ""
