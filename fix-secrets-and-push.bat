@echo off
REM Fix: Remove exposed secrets from git history and push clean version

echo ========================================================================
echo FIXING EXPOSED SECRETS IN GIT HISTORY
echo ========================================================================
echo.

echo Step 1: Removing last commit with secrets...
git reset --soft HEAD~1
echo Done!
echo.

echo Step 2: Staging cleaned files...
git add CLOUDINARY_INTEGRATION_COMPLETE.md
git add COMPLETE_DEPLOYMENT_GUIDE.md
git add DEPLOY_NOW.md
git add .
echo Done!
echo.

echo Step 3: Creating new commit (without secrets)...
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
echo Done!
echo.

echo Step 4: Force pushing to HuggingFace...
git push origin main --force
echo.

echo ========================================================================
echo COMPLETE!
echo ========================================================================
echo.
echo Next steps:
echo 1. Check if push succeeded
echo 2. Add secrets manually in HF Space Settings (Repository secrets)
echo 3. IMPORTANT: Regenerate any exposed tokens for security!
echo.
pause
