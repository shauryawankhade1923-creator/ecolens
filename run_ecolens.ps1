Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "  🌲 EcoLens 2.0 (FastAPI + Tailwind Web Application)" -ForegroundColor Green
Write-Host "========================================================`n" -ForegroundColor Cyan
Write-Host "Starting server at http://localhost:8000 ...`n" -ForegroundColor Yellow

Start-Process "http://localhost:8000"
py -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
