@echo off
echo ========================================================
echo Launching On-Device AI Reliability Monitor (Desktop)
echo ========================================================

start cmd /k "python scripts/start_backend.py"
start cmd /k "cd frontend && npm run dev"

echo Backend started at:  http://127.0.0.1:8000
echo Frontend dev server: http://127.0.0.1:5173
pause
