# YOLOv8 机器人场景目标检测项目

**作者**：卢欣怡 (计工本2603班, 学号 202611160325)

## 📌 项目简介
本项目为计算机与人工智能学院机器人实验室 2026 年工程实践能力考核的提交成果。采用 YOLOv8 模型，对可乐、障碍物、足球三类目标进行检测。详细实验过程见 `report.md`。

## 📂 文件结构
*   `train.py` : 模型训练脚本
*   `predict.py` : 模型推理预测脚本
*   `data.yaml` : 数据集与类别配置文件
*   `requirements.txt` : 依赖库清单
*   `best.pt` : 训练好的最优模型权重
*   `dataset/` : 部分数据集样本 (图片与标签)

## 🚀 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 开始训练：`python train.py`
3. 模型预测：`python predict.py`