@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [12] batch_01 / TRAIN part (34 images) - dedicated py3.11 env ===
echo Images : label_batches\_stage\b01_train   (only batch_01 images, no scrolling)
echo Save to: dataset\labels\train
echo.
echo Step 1: click YOLO in the left toolbar (save format)
echo Step 2: draw obstacle boxes, press Ctrl+S for every image
echo Step 3: when done, run: check_transfer.py  (count should go from 0 up)
echo.
"C:\Users\dell\label-env\Scripts\labelImg.exe" "%~dp0label_batches\_stage\b01_train" "%~dp0classes.txt" "%~dp0dataset\labels\train"
if "%~1"=="" pause
