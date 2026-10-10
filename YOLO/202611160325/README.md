# YOLOv8 机器人场景目标检测项目

作者：卢欣怡 (计工本2603班, 学号 202611160325)

 项目简介
本项目为2026年工程实践能力考核的提交成果。采用 YOLOv8 模型，对可乐（cola）、障碍物（obstacle）、足球（football）三类目标进行检测。

详细的实验过程、数据分析、不足与反思，请查阅 **[report.md](./report.md)**。

文件结构
* `train.py`：模型训练脚本
* `predict.py`：模型推理脚本
* `data.yaml`：数据集与类别配置文件
* `requirements.txt`：依赖库清单
* `dataset/`：部分数据集样本（包含图片与对应标签）
* `runs/`：训练过程截图、环境配置截图、模型训练结果曲线图
* `results/`：模型预测结果图片（包含目标框、类别及置信度）
* `report.md`：完整的详细实验报告

*(注：由于GitHub文件大小限制，训练生成的模型权重文件 `best.pt` 未直接上传，运行 `train.py` 即可生成。)*

 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 开始训练：`python train.py`
3. 模型预测：`python predict.py`