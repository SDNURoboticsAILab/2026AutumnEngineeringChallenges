# 基于 YOLO 的机器人场景目标检测

> 计工本2602　王喻晨　202611160208

使用实验室提供的 obstacle / cola / football 三类机器人场景图片（共 935 张），完成了数据整理、目标标注、YOLO 模型训练与新图片目标检测的完整流程，并用 Streamlit 封装了本地检测页面。完整过程与实验记录见 [report.md](report.md)。

## 运行环境

- Windows 11，Python 3.11（Miniconda 虚拟环境 `yolo`）
- PyTorch 2.11.0+cu128，ultralytics 8.4.155（完整版本见 [requirements.txt](requirements.txt)）
- RTX 50 系显卡需 cu128 轮子，安装细节见 report.md「环境配置」一节

## 快速开始

激活环境：`conda activate yolo`。

| 操作     | 方式                                                           |
| ------ | ------------------------------------------------------------ |
| 训练     | `python train.py --name exp1`                                |
| 检测新图片  | 图片放入 `new_images/`，运行 `python predict.py`（`--conf 0.4` 可调阈值） |
| 本地检测页面 | `streamlit run app.py`，浏览器打开 http://localhost:8501           |

## 说明

- 最终模型权重：`runs/fixed_v4/weights/best.pt`（mAP50 = 0.888，已随仓库提交，克隆后可直接推理）
- `data.yaml` 中 `path` 使用相对路径（`dataset`），无需修改即可运行
- 数据集图片未重复提交（原始图片见仓库根目录 `obstacle/cola/football`），本目录保留完整 `dataset/labels` 标签与 `split_manifest.csv` 划分清单，按 report.md「数据集整理」一节可完整复现数据集
- 预标注/训练脚本需要预训练权重 `downloads/yolov8s-worldv2.pt`、`downloads/yolov8s.pt`（重量级文件未入库，下载方式见 report.md 问题 #5）

## 目录结构

```
202611160208/
├── train.py / predict.py / app.py / data.yaml   # 训练/推理/页面代码与数据集配置
├── dataset/labels/   # YOLO 标签（train/val，939 个）
├── runs/             # fixed_v4 最终实验 + exp_20261004_1813 实验记录（最终权重 fixed_v4/weights/best.pt）
├── results/          # 检测输出图 + detections.csv
├── new_images/       # 测试图
├── screenshots/      # Level 1~5 验证截图
├── predefined_classes.txt / split_manifest.csv   # 标注类别 / 数据划分清单
└── requirements.txt / README.md / report.md
```
