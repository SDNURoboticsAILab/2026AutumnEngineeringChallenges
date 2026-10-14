@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo ============================================================
echo  labelImg 调试启动（把报错完整写进日志文件）
echo  日志: %~dp0_labelimg_error.log
echo ============================================================
echo.
echo 注意：本文件会自动带上"图片目录 / 类别文件 / 标签保存目录"三个参数。
echo       不要直接双击 labelImg.exe —— 那样它会弹目录选择框，取消就退出。
echo.
"C:\Users\dell\yolo-env\Scripts\labelImg.exe" "%~dp0dataset\images\train" "%~dp0classes.txt" "%~dp0dataset\labels\train" > "%~dp0_labelimg_error.log" 2>&1
echo.
echo ==== 程序已退出，退出码: %ERRORLEVEL% ====
echo ==== 下面是从日志里读出的内容（为空 = 正常启动过）====
type "%~dp0_labelimg_error.log"
echo.
echo ==== 日志文件位置: %~dp0_labelimg_error.log ====
pause
