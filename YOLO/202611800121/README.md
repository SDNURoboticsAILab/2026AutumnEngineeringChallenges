# 基于 YOLO 的机器人场景目标检测

## 项目简介
本项目基于 Ultralytics YOLO 实现 obstacle（障碍物）、cola（可乐）、football（足球）三类目标的检测。

## 环境配置
- Python 3.11
- PyTorch 2.14.1
- Ultralytics 8.4.174
- 操作系统：Windows

## 数据集
原始数据共 949 张图片（obstacle: 316, cola: 293, football: 340），按 80% / 20% 划分为训练集（759张）和验证集（190张）。
类别编号：0=obstacle, 1=cola, 2=football

## 训练
使用 YOLO11n 模型，imgsz=640，batch=8，共训练 79 轮，mAP50 稳定在 0.97 后手动终止。
最终模型权重：`runs/detect/runs/train/weights/best.pt`

## 推理
使用 `predict.py` 对训练集之外的新图片进行检测，结果见 `results/` 文件夹。

## 目录说明
- `train.py`：训练脚本
- `predict.py`：推理脚本
- `data.yaml`：数据集配置
- `classes.txt`：类别名称文件
- `results/`：新图片检测结果
- `项目截图/`：各 Level 验证截图

> 数据集因体积过大未上传，如需复现请联系作者。