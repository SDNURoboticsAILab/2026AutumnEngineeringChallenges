@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [3] Labeling progress ===
"C:\Users\dell\yolo-env\Scripts\python.exe" batch_progress.py
echo.
echo === placement / obstacle box count ===
"C:\Users\dell\yolo-env\Scripts\python.exe" check_transfer.py
if "%~1"=="" pause
