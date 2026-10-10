# 基于 YOLO 的机器人场景目标检测

本项目围绕 2026 年全国大学生计算机系统能力大赛智能系统创新设计赛（小米杯）相关机器人场景展开，使用 Ultralytics YOLOv8 完成了 obstacle、cola、football 三类目标的数据整理、标注、模型训练以及推理应用。

## 🛠️ 环境配置
- **操作系统**：Windows 10/11
- **Python 版本**：Python 3.12.9
- **深度学习框架**：PyTorch 2.14.1 (CPU版本)
- **目标检测框架**：Ultralytics YOLOv8 (8.4.175)
- **其他依赖**：OpenCV-Python, Streamlit

**安装依赖：**
```bash
pip install ultralytics opencv-python streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple