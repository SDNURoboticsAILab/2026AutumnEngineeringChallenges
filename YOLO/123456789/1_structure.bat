@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [1] Directory structure compliance check (Level 1) ===
"C:\Users\dell\yolo-env\Scripts\python.exe" verify_structure.py
echo.
echo === project tree ===
"C:\Users\dell\yolo-env\Scripts\python.exe" show_structure.py
if "%~1"=="" pause
