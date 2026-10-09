# 基于YOLO的机器人场景目标检测项目

## 项目简介
本项目为2026秋季工程实践挑战赛（YOLO题目）提交作品。
使用 Ultralytics YOLO 模型，针对特定机器人场景进行目标检测训练与推理。
**检测类别**：
- [0] 可乐 (cola)
- [1] 足球 (football)
- [2] 障碍物 (obstacle)

**作者**：[何安澜]  
**学号**：[202611160503]  
**班级**：[计工本2605]

## 环境依赖
项目基于以下环境开发与测试，完整依赖见 `requirements.txt`：
- Python: 3.11.16
- PyTorch: 2.5.0 + CUDA 11.8
- Ultralytics: 8.3.163 

## 目录结构
```text
202611160503/
├── README.md           # 本文件
├── train.py            # 模型训练脚本
├── predict.py          # 模型推理脚本
├── data.yaml           # 数据集配置文件
├── requirements.txt    # 依赖环境清单
├── dataset/            # 数据集目录（如需体验，请从网盘下载）
├── runs/               # 训练输出目录（含权重和曲线图）
├── results/            # 训练结果与验证截图存放处
└── report.md           # 详细的工程实践报告
