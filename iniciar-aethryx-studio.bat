@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno .venv de AETHRYX Studio.
    exit /b 1
)
if /I "%~1"=="--diagnostico" (
    ".venv\Scripts\python.exe" -m aethryx_preflight
    exit /b
)
start "" ".venv\Scripts\pythonw.exe" app.py
