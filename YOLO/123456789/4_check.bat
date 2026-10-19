@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [4] Label format check ===
"C:\Users\dell\yolo-env\Scripts\python.exe" check_labels.py
echo.
echo === Label quality audit (missing / duplicate / conflict) ===
"C:\Users\dell\yolo-env\Scripts\python.exe" audit_labels.py
if "%~1"=="" pause
