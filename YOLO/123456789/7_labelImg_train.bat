@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [7] labelImg - annotate TRAIN set ===
echo Images : dataset\images\train
echo Classes: classes.txt  (0 obstacle / 1 cola / 2 football)
echo Save to: dataset\labels\train   (YOLO txt written directly here)
echo.
echo Remember: switch the save format to YOLO in the left toolbar first!
echo.
"C:\Users\dell\yolo-env\Scripts\labelImg.exe" "%~dp0dataset\images\train" "%~dp0classes.txt" "%~dp0dataset\labels\train"
if "%~1"=="" pause
