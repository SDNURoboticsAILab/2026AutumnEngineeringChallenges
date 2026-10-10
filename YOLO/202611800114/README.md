# 基于 YOLO 的机器人场景目标检测

本项目使用 YOLO11n 对 obstacle、cola、football 三类目标进行检测，完整流程包括数据集整理、目标标注、模型训练和新图片推理。

## 目录说明

- `dataset/`：数据集（images + labels）
- `train.py`：模型训练脚本
- `predict.py`：新图片推理脚本
- `data.yaml`：数据集配置文件
- `runs/train/`：训练结果与模型权重
- `results/predict/`：推理检测结果图
- `screenshots/`：各 Level 验证截图
- `report.md`：完整项目报告

## 数据集

- 类别：0-obstacle, 1-cola, 2-football
- 训练集 300 张，验证集 60 张

## 训练结果

- mAP50 = 0.962
- 模型文件：runs/train/weights/best.pt

## 详细报告

见 [report.md](report.md)