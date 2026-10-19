@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [10] labelImg (python 3.11 dedicated env) - TRAIN set ===
echo Env    : C:\Users\dell\label-env   (Python 3.11.9 + PyQt5 5.15.9)
echo Images : dataset\images\train
echo Classes: classes.txt  (0 obstacle / 1 cola / 2 football)
echo Save to: dataset\labels\train
echo.
echo Remember: click YOLO in the left toolbar before drawing boxes.
echo.
"C:\Users\dell\label-env\Scripts\labelImg.exe" "%~dp0dataset\images\train" "%~dp0classes.txt" "%~dp0dataset\labels\train"
if "%~1"=="" pause
