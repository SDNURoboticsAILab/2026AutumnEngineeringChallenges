@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo ============================================================
echo  实时监视标注：只要有一个 obstacle 框落盘就会立刻打印
echo  标注时把这个窗口挂在旁边即可；Ctrl+C 退出
echo ============================================================
echo.
"C:\Users\dell\yolo-env\Scripts\python.exe" -u watch_labels.py
pause
