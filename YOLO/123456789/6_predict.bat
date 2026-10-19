@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [6] Predict on new images (Level 4) ===
echo Put new images into test_images\ first.
"C:\Users\dell\yolo-env\Scripts\python.exe" predict.py
echo.
echo Results: results\predict\
if "%~1"=="" pause
