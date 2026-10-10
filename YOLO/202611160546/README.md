# 基于 YOLO 的机器人场景目标检测

> 计工本2605　于子涵　202611160546

使用实验室提供的 `obstacle` / `cola` / `football` 三类机器人场景图片（共 949 张），
完成数据整理、目标标注、YOLO 模型训练与新图片目标检测的完整流程，
并用 Streamlit 封装了本地检测页面。完整过程记录见 [report.md](report.md)。

## 类别定义

| 编号 | 名称       | 视觉所指                     |
| -- | -------- | ------------------------ |
| 0  | obstacle | 画面中的**蓝色泡沫障碍箱**          |
| 1  | cola     | **黑色可乐瓶**（瓶身近黑，有塑料高光）    |
| 2  | football | **黑白相间的足球**（场景中的橙色瑜伽球**不算**） |

## 运行环境

- Windows 11，Python 3.14
- PyTorch 2.14.1+cpu，ultralytics 8.4.175
- **本机显卡为 AMD RX 7700 XT，无 CUDA，训练在 CPU（16 逻辑核）上进行**

完整依赖版本见 [requirements.txt](requirements.txt)。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 用实验室提供的原始图片重建数据集（原始图片位于仓库 YOLO/obstacle、cola、football）
python prepare_dataset.py --src-root ..

# 2) （可选）重新生成预标注；仓库中已附带标注好的 dataset/labels，可直接跳过
python tools/pre_annotate.py --weights ../weights/yolov8s.pt --dataset dataset

# 3) 训练
python train.py --name exp1

# 4) 检测新图片（图片放进 new_images/）
python predict.py --conf 0.25

# 5) 本地检测页面
streamlit run app.py      # 浏览器打开 http://localhost:8501
```

## 目录结构

```
202611160546/
├── README.md / report.md          # 说明与项目报告
├── requirements.txt
├── data.yaml                      # 数据集配置（相对路径，克隆后可直接训练）
├── prepare_dataset.py             # 由原始图片重建 dataset/（划分 train/val）
├── train.py                       # 训练脚本
├── predict.py                     # 新图片推理脚本（输出结果图 + detections.csv）
├── app.py                         # Level 5 本地识别页面（Streamlit）
├── tools/
│   ├── pre_annotate.py            # 半自动预标注流水线
│   └── inspect_labels.py          # 标注复核工具（把标签画回图片拼图）
├── dataset/
│   ├── images/{train,val}/        # 图片（原始图片为实验室提供，未重复入库）
│   └── labels/{train,val}/        # YOLO 标签
├── split_manifest.csv             # 训练/验证划分清单
├── runs/                          # 训练结果（含最终权重 best.pt）
├── results/                       # 新图片检测结果与 detections.csv
├── new_images/                    # 待检测的新图片
└── screenshots/                   # 各 Level 验证截图
```

## 说明

- **数据集**：共 949 张（obstacle 316 / cola 293 / football 340），
  按类别内部 8:2 随机划分，固定随机种子 42，得到 train 759 / val 190。
  划分清单见 `split_manifest.csv`。
- **标注方式**：采用"半自动预标注 + 人工抽查复核"。
  蓝色障碍箱用 HSV 颜色分割；可乐瓶与足球用 COCO 预训练检测器配合颜色仲裁，
  其中推理分辨率按类别分别设置（详见 report.md「目标标注」一节）。
- **图片未入库**：原始图片体积较大，仓库中只保留标签与划分清单，
  按上面第 1 步可完整重建数据集。
- **模型权重**：最终权重为 `runs/exp1/weights/best.pt`，随仓库提交，克隆后可直接推理。
