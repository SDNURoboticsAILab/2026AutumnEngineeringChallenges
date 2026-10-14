@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [13] batch_01 / VAL part (6 images) - dedicated py3.11 env ===
echo Images : label_batches\_stage\b01_val
echo Save to: dataset\labels\val
echo.
echo Step 1: click YOLO in the left toolbar (save format)
echo Step 2: draw obstacle boxes, press Ctrl+S for every image
echo Step 3: when done, run: check_transfer.py
echo.
"C:\Users\dell\label-env\Scripts\labelImg.exe" "%~dp0label_batches\_stage\b01_val" "%~dp0classes.txt" "%~dp0dataset\labels\val"
if "%~1"=="" pause
