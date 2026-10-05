# 基于 YOLO 的机器人场景目标检测

> 计工本2602　王喻晨　202611160208

使用实验室提供的 obstacle / cola / football 三类机器人场景图片（共 935 张），完成了数据整理、目标标注、YOLO 模型训练与新图片目标检测的完整流程，并用 Streamlit 封装了本地检测页面。完整过程与实验记录见 [report.md](report.md)。

## 运行环境

- Windows 11，Python 3.11（Miniconda 虚拟环境 `yolo`）
- PyTorch 2.11.0+cu128，ultralytics 8.4.155（完整版本见 [requirements.txt](requirements.txt)）
- RTX 50 系显卡需 cu128 轮子，安装细节见 report.md「环境配置」一节

## 快速开始

激活环境：`conda activate yolo`（或直接双击下方 bat，默认调用 `%USERPROFILE%\miniconda3\envs\yolo`）。

| 操作     | 方式                                                        |
| ------ | --------------------------------------------------------- |
| 训练     | `python yolo_project/train.py --name exp1` 或双击 `启动训练.bat` |
| 检测新图片  | 图片放入 `new_images/`，双击 `启动检测.bat`                          |
| 本地检测页面 | 双击 `启动页面.bat`，浏览器自动打开 http://localhost:8501               |
| 标注修正   | 双击 `启动标注.bat`（val 集），传参 train 处理训练集                       |

## 说明

- 最终模型权重：`yolo_project/runs/fixed_v4/weights/best.pt`（mAP50 = 0.888）
- `yolo_project/data.yaml` 中 `path` 为本机绝对路径，**换机运行前请改成本机的 dataset 目录**
- 预标注/训练脚本需要预训练权重 `downloads/yolov8s-worldv2.pt`、`downloads/yolov8s.pt`（重量级文件未入库，下载方式见 report.md 问题 #5）

## 目录结构

```
yolo_project/          # 训练/推理/页面代码 + 数据集 + 实验记录
scripts/               # 数据体检/划分/预标注/清洗/质检/标注器等 11 个脚本
new_images/            # 训练集之外的测试图
screenshots/           # Level 1~5 验证截图
report.md              # 项目报告
```
