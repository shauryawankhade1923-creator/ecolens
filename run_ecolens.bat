@echo off
title EcoLens 2.0 - Environmental Intelligence Platform
echo.
echo ========================================================
echo   🌲 EcoLens 2.0 (FastAPI + Tailwind Web Application)
echo ========================================================
echo.
echo Starting EcoLens server at http://localhost:8000 ...
echo.

start "" "http://localhost:8000"
py -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
if %ERRORLEVEL% NEQ 0 (
    python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
)
pause
