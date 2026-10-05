@echo off
rem 检测 new_images\ 里的图片（先把新图片放进去；可传参改置信度阈值，如 启动检测.bat 0.4）
cd /d %~dp0
if "%1"=="" (
  "%USERPROFILE%\miniconda3\envs\yolo\python.exe" "yolo_project\predict.py" --source "new_images"
) else (
  "%USERPROFILE%\miniconda3\envs\yolo\python.exe" "yolo_project\predict.py" --source "new_images" --conf %1
)
pause
