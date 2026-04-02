@echo off
REM ============================================================
REM Docker Run Script for MAdVerse (AdCraft AI) - Windows
REM Usage: docker-run.bat
REM ============================================================

echo Building MAdVerse Docker image...
docker build -t madverse-app .

echo.
echo Starting MAdVerse container...
docker run -d ^
  --name madverse-app ^
  -p 8000:8000 ^
  --env-file .env ^
  -v "%cd%\products_db:/app/products_db" ^
  -v "%cd%\outputs:/app/outputs" ^
  -v "%cd%\uploads:/app/uploads" ^
  --restart unless-stopped ^
  madverse-app

echo.
echo ✅ MAdVerse is starting...
echo 📍 Access at: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo View logs: docker logs -f madverse-app
echo Stop: docker stop madverse-app
echo Remove: docker rm madverse-app
