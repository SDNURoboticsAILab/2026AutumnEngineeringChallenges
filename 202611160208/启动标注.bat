@echo off
rem 打开标注工具（默认 val 集；传参 train 处理训练集）
cd /d %~dp0
set SPLIT=%1
if "%SPLIT%"=="" set SPLIT=val
start "" "%USERPROFILE%\miniconda3\envs\yolo\pythonw.exe" "scripts\annotator.py" --split %SPLIT%
