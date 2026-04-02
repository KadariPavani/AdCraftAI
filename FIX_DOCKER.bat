@echo off
REM ============================================================
REM Fix Docker Line Ending Issues for MAdVerse
REM ============================================================

echo Fixing start.sh line endings...

REM Stop and remove existing container
docker-compose down 2>nul

REM Convert line endings using PowerShell
powershell -Command "(Get-Content scripts\start.sh -Raw) -replace \"`r`n\", \"`n\" | Set-Content -NoNewline scripts\start.sh"

echo Done! Line endings fixed.
echo.
echo Now run: docker-compose up -d --build
pause
