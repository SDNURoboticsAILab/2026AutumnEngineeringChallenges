@echo off
rem 启动本地检测页面（Streamlit），就绪后浏览器自动打开 http://localhost:8501
cd /d %~dp0
"%USERPROFILE%\miniconda3\envs\yolo\Scripts\streamlit.exe" run "yolo_project\app.py"
