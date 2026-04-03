@echo off
REM ============================================================
REM Quick Deployment Script for Hugging Face Spaces (Windows)
REM ============================================================

echo ================================================
echo   MAdVerse - Hugging Face Spaces Deployment
echo ================================================
echo.

REM Check if Git LFS is installed
where git-lfs >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git LFS is not installed!
    echo.
    echo Please install Git LFS first:
    echo   Download from https://git-lfs.github.com/
    echo.
    pause
    exit /b 1
)

echo [OK] Git LFS is installed

REM Check if .gitattributes exists
if not exist ".gitattributes" (
    echo ERROR: .gitattributes file not found!
    echo   This file is required for Git LFS tracking.
    pause
    exit /b 1
)

echo [OK] .gitattributes found

REM Verify FAISS index
echo.
echo [VALIDATION] Checking FAISS index...
if not exist "embeddings\faiss_indexes\madverse_index.faiss" (
    echo ERROR: FAISS index not found!
    echo   Run 'git lfs pull' to download LFS files.
    pause
    exit /b 1
)

REM Check file size (basic check)
for %%A in ("embeddings\faiss_indexes\madverse_index.faiss") do set FAISS_SIZE=%%~zA
if %FAISS_SIZE% LSS 100000000 (
    echo ERROR: FAISS index appears to be a Git LFS pointer!
    echo   Size: %FAISS_SIZE% bytes (expected ^> 100 MB^)
    echo.
    echo   Fix: Run 'git lfs pull' to download actual files.
    pause
    exit /b 1
)

set /a FAISS_MB=%FAISS_SIZE% / 1048576
echo [OK] FAISS index OK (%FAISS_MB% MB)

REM Check if HF Space URL is provided
if "%~1"=="" (
    echo.
    echo ERROR: Hugging Face Space URL required!
    echo.
    echo Usage:
    echo   scripts\deploy-to-hf.bat https://huggingface.co/spaces/YOUR_USERNAME/madverse
    echo.
    echo Steps:
    echo   1. Create a Space at https://huggingface.co/new-space
    echo   2. Choose SDK: Docker, Hardware: cpu-basic
    echo   3. Copy the Space URL
    echo   4. Run this script with the URL
    echo.
    pause
    exit /b 1
)

set HF_SPACE_URL=%~1
set TEMP_DIR=hf_deploy_temp

echo.
echo ================================================
echo   Deployment Target: %HF_SPACE_URL%
echo ================================================
echo.

REM Ask for confirmation
set /p CONFIRM="Deploy to this Space? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Deployment cancelled.
    pause
    exit /b 0
)

REM Clone HF Space
echo.
echo [STEP 1/5] Cloning Hugging Face Space...
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
git clone "%HF_SPACE_URL%" "%TEMP_DIR%"
cd "%TEMP_DIR%"

REM Copy files
echo.
echo [STEP 2/5] Copying project files...
echo   Excluding: .git, .venv, data\images, outputs, uploads, products_db

xcopy /E /I /Y /Q ..\app app\
xcopy /E /I /Y /Q ..\scripts scripts\
xcopy /E /I /Y /Q ..\embeddings embeddings\
xcopy /E /I /Y /Q ..\processed processed\
xcopy /E /I /Y /Q ..\data\annotations data\annotations\
xcopy /E /I /Y /Q ..\data\indices data\indices\
copy /Y ..\*.py .
copy /Y ..\*.md .
copy /Y ..\*.txt .
copy /Y ..\*.yml .
copy /Y ..\*.yaml .
copy /Y ..\Dockerfile .
copy /Y ..\.gitattributes .
copy /Y ..\.gitignore .
copy /Y ..\.dockerignore .
copy /Y ..\.env.example .

REM Use HF Space README
echo.
echo [STEP 3/5] Setting up README for Hugging Face...
if exist "README_HF_SPACE.md" (
    copy /Y README.md README_GITHUB.md >nul 2>&1
    copy /Y README_HF_SPACE.md README.md
    echo   [OK] README.md updated for Hugging Face Spaces
) else (
    echo   [WARNING] README_HF_SPACE.md not found, using existing README.md
)

REM Setup Git LFS
echo.
echo [STEP 4/5] Configuring Git LFS...
git lfs install

git lfs track "embeddings/image_embeddings.pkl"
git lfs track "embeddings/faiss_indexes/*.faiss"
git lfs track "embeddings/faiss_indexes/*.pkl"
git lfs track "processed/metadata/*.csv"
git lfs track "data/annotations/*.json"

echo   [OK] Git LFS configured

REM Commit and push
echo.
echo [STEP 5/5] Committing and pushing to Hugging Face...
git add .
git add .gitattributes

git commit -m "Deploy MAdVerse AI to Hugging Face Spaces - Features: FastAPI ad generation, FAISS search (50K+ ads), Multi-language support, Docker deployment, cpu-basic optimized"

echo.
echo   Pushing to Hugging Face (this may take 5-10 minutes for LFS files)...
git push origin main

echo.
echo ================================================
echo   Deployment Complete!
echo ================================================
echo.
echo Next steps:
echo.
echo 1. Add API keys in Space Settings:
echo    %HF_SPACE_URL%/settings
echo.
echo    Required (at least one):
echo    - GOOGLE_API_KEY  (Google Gemini)
echo    - GROQ_API_KEY    (Groq)
echo.
echo    Optional (better quality):
echo    - HF_TOKEN        (Hugging Face)
echo    - TOGETHER_API_KEY (Together AI)
echo.
echo 2. Monitor build progress:
echo    %HF_SPACE_URL%/logs
echo.
echo 3. Build duration: ~10-15 minutes (first time)
echo.
echo 4. Once running, your Space will be available at:
echo    %HF_SPACE_URL:/spaces/=%
echo.
echo ================================================
echo.

REM Cleanup
cd ..
set /p CLEANUP="Delete temporary deployment folder? (y/n): "
if /i "%CLEANUP%"=="y" (
    rmdir /s /q "%TEMP_DIR%"
    echo [OK] Cleanup complete
)

echo.
echo Happy deploying!
pause
