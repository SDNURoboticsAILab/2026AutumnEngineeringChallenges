@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [8] labelImg - annotate VAL set ===
echo Images : dataset\images\val
echo Classes: classes.txt  (0 obstacle / 1 cola / 2 football)
echo Save to: dataset\labels\val
echo.
echo Remember: switch the save format to YOLO in the left toolbar first!
echo.
"C:\Users\dell\yolo-env\Scripts\labelImg.exe" "%~dp0dataset\images\val" "%~dp0classes.txt" "%~dp0dataset\labels\val"
if "%~1"=="" pause
