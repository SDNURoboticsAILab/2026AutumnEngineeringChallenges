@echo off
rem 启动训练，用法：启动训练.bat [实验名] [轮数]（默认 yolov8s、100 轮、batch 16）
cd /d %~dp0
set PY=%USERPROFILE%\miniconda3\envs\yolo\python.exe
set NAME=%1
if "%NAME%"=="" set NAME=exp_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%
set NAME=%NAME: =0%
set EPOCHS=%2
if "%EPOCHS%"=="" set EPOCHS=100
"%PY%" "yolo_project\train.py" --model "downloads\yolov8s.pt" --epochs %EPOCHS% --batch 16 --name %NAME%
