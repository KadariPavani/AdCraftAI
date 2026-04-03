@echo off
REM ============================================================
REM Quick Setup Script for Cloud Storage Migration
REM ============================================================

echo.
echo ============================================================
echo   MAdVerse - Cloud Storage Setup
echo ============================================================
echo.

REM Step 1: Install dependencies
echo [1/5] Installing cloud storage dependencies...
pip install cloudinary psycopg2-binary --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo   ✓ cloudinary installed
echo   ✓ psycopg2-binary installed
echo.

REM Step 2: Create storage directory
echo [2/5] Creating app/storage directory...
if not exist "app\storage" (
    mkdir "app\storage"
    echo   ✓ Created app\storage\
) else (
    echo   ✓ app\storage\ already exists
)
echo.

REM Step 3: Copy storage module files
echo [3/5] Setting up storage module...

REM Create __init__.py
echo """Storage module""" > "app\storage\__init__.py"
echo from .cloudinary_manager import CloudinaryManager, get_cloudinary_manager >> "app\storage\__init__.py"
echo from .image_storage import ImageStorage >> "app\storage\__init__.py"
echo __all__ = ['CloudinaryManager', 'get_cloudinary_manager', 'ImageStorage'] >> "app\storage\__init__.py"
echo   ✓ Created app\storage\__init__.py

REM Copy cloudinary_manager.py
copy /Y "CLOUD_SETUP_cloudinary_manager.py" "app\storage\cloudinary_manager.py" >nul
echo   ✓ Created app\storage\cloudinary_manager.py

REM Copy image_storage.py  
copy /Y "CLOUD_SETUP_image_storage.py" "app\storage\image_storage.py" >nul
echo   ✓ Created app\storage\image_storage.py
echo.

REM Step 4: Check for .env file
echo [4/5] Checking .env configuration...
if exist ".env" (
    echo   ✓ .env file found
    echo.
    echo   ⚠️  IMPORTANT: Add these lines to your .env file:
    echo.
    echo   CLOUDINARY_CLOUD_NAME=your-cloud-name
    echo   CLOUDINARY_API_KEY=your-api-key
    echo   CLOUDINARY_API_SECRET=your-api-secret
    echo   USE_CLOUD_STORAGE=true
    echo.
) else (
    echo   ⚠️  .env file not found!
    echo   Creating from example...
    copy /Y ".env.cloudinary.example" ".env" >nul
    echo   ✓ Created .env - PLEASE EDIT WITH YOUR CREDENTIALS
    echo.
)

REM Step 5: Final instructions
echo [5/5] Setup complete!
echo.
echo ============================================================
echo   Next Steps:
echo ============================================================
echo.
echo 1. Sign up for Cloudinary (free):
echo    https://cloudinary.com/users/register/free
echo.
echo 2. Get your credentials from the dashboard
echo.
echo 3. Edit .env file and add:
echo    CLOUDINARY_CLOUD_NAME=your-cloud-name
echo    CLOUDINARY_API_KEY=your-api-key
echo    CLOUDINARY_API_SECRET=your-api-secret
echo    USE_CLOUD_STORAGE=true
echo.
echo 4. Test the integration:
echo    python run.py
echo.
echo 5. Check logs for:
echo    [CLOUDINARY] ✅ Configured
echo    [STORAGE] ✅ Using Cloudinary
echo.
echo ============================================================
echo   Read CLOUD_SETUP_INSTRUCTIONS.md for details
echo ============================================================
echo.

pause
