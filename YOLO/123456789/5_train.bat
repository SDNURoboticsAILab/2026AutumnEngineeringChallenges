@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [5] Train YOLO model ===
echo (preflight will stop early if some class has no labels)
"C:\Users\dell\yolo-env\Scripts\python.exe" train.py
if "%~1"=="" pause
