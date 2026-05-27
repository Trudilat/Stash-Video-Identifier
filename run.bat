@echo off
cd /d "%~dp0"

:: ── Install dependencies if rich is missing ───────────────────────────────
python -c "import rich" 2>nul || (
    echo Installing dependencies...
    pip install -r requirements.txt
)

:: ── Check Ollama is running ────────────────────────────────────────────────
tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if errorlevel 1 (
    echo Ollama is not running. Please start it before launching Video Identifier.
    pause
    exit /b 1
)

python app\main.py
pause
