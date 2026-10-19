@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === [2] Start labeling tool (batch_01) ===
echo A new window will run the server; browser should open automatically.
start "label_tool" "C:\Users\dell\yolo-env\Scripts\python.exe" label_tool.py --batch label_batches\batch_01.txt
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8888
echo.
echo Browser URL: http://127.0.0.1:8888
echo To stop: close the "label_tool" window, or press Ctrl+C there.
if "%~1"=="" pause
