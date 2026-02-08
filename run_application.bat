@echo off
call conda activate itemmem

start "API" cmd /k uvicorn app.api.server:app
timeout /t 2 >nul

start "INGEST" cmd /k python scripts/ingest_ipcam.py
timeout /t 2 >nul

start "UI" cmd /k streamlit run app/ui/streamlit_app.py
