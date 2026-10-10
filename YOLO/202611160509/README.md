# YOLO11 目标检测项目

基于 **YOLO11n** 的三目标（cola / football / obstacle）检测：训练 → 自动标注 → 推理 → 验证，端到端可复现。

## 目录结构

```
├── README.md            # 项目说明（本文件）
├── report.md            # 项目报告（环境/数据/标注/训练/结果/问题）
├── requirements.txt     # 依赖清单与安装命令
├── train.py             # 模型训练脚本
├── predict.py           # 模型推理/检测脚本
├── auto_label.py        # 自动标注脚本（生成标准数据集）
├── val.py               # 模型验证脚本
├── data.yaml            # 数据集配置
├── dataset/             # 人工标注数据集（images/labels，train 66 / val 28）
├── dataset_auto/        # 自动标注数据集（train 678 / val 169）
├── runs/
│   ├── weights/         # yolo11n.pt（预训练）/ best.pt / last.pt / 权重信息.txt
│   ├── train/           # results.png、results.csv、混淆矩阵、各曲线
│   └── val/             # 验证输出
└── results/             # 训练结果截图、检测结果截图、过程记录（level1~4）
```

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.11（Anaconda 环境） |
| torch | 2.11.0+cu128（CUDA 版，必须） |
| torchvision | 0.26.0+cu128 |
| ultralytics | 8.3.163 |
| GPU | NVIDIA RTX 5060（8GB，CUDA 可用） |

安装：见 `requirements.txt`。

## 数据集

3 个类别：`cola`（可乐）、`football`（足球）、`obstacle`（障碍物）。

- `dataset/`：LabelImg 人工标注，66 训练 / 28 验证，YOLO 格式
- `dataset_auto/`：best.pt 自动标注，678 训练 / 169 验证（80/20）

> **关于本仓库内的数据内容：** 为控制提交体积，`images/` 下的原始图片（`dataset/images` 94 张、`dataset_auto/images` 847 张，约 107 MB）
> 未纳入本次提交，但**标签文件（`labels/`）与数据集配置（`data.yaml`）均已完整提交**。
> 四个划分的标签数量与图片数量严格一一对应：
> `dataset` train 66 / val 28，`dataset_auto` train 678 / val 169（每张图一个同名 `.txt`，无目标图片为空文件）。
> 如需复现训练，把实验室提供的 `obstacle`、`cola`、`football` 原始图片按同名文件放回 `images/` 对应划分即可，
> `data.yaml` 使用相对路径，脚本无需任何修改。训练结果、权重与检测效果见 `results/`。

## 使用方法

```bash
# 1) 训练（默认 50 epochs / batch 16 / imgsz 640；读取根目录 data.yaml → dataset/）
python train.py
python train.py --epochs 100 --batch 8

# 2) 推理/检测（输出框+类别+置信度效果图）
python predict.py --source 待检测图片文件夹
python predict.py --source 图片.jpg --conf 0.3 --save-txt

# 3) 自动标注（生成标准数据集）
python auto_label.py --source 未标注图片文件夹

# 4) 验证（输出 mAP50 / P / R 及各类别指标；读取根目录 data.yaml）
python val.py
```

## 训练结果（best.pt，验证集）

| 指标 | 值 |
|---|---|
| mAP50 | 0.9886 |
| mAP50-95 | 0.8933 |
| Precision | 0.9439 |
| Recall | 0.9850 |

| 类别 | P | R | mAP50 |
|---|---|---|---|
| cola | 0.937 | 1.000 | 0.995 |
| football | 0.900 | 0.998 | 0.978 |
| obstacle | 0.995 | 0.957 | 0.993 |

## 注意事项

1. torch 必须装 CUDA 版（cu128 构建），否则无法使用 GPU；RTX 50 系列需 cu128+。
2. Windows 下训练 workers 必须为 0（多进程会触发 CUDA illegal memory access），脚本已固定。
3. 无 GPU 时：训练/推理将 `device=0` 改为 `device="cpu"`。

详细过程见 `report.md` 与 `results/过程记录/`。
