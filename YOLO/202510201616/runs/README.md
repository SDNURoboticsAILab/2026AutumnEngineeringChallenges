# runs / 训练结果

## 训练目录

```text
runs/train_20261018_1701/
```

## 训练参数

| 参数 | 值 |
|---|---|
| 模型 | `yolo11n.pt`（首次运行自动下载） |
| 数据 | `data.yaml`（751 张，601 train / 150 val） |
| epochs | 50 |
| imgsz | 640 |
| batch | 16 |
| device | GPU 0（NVIDIA RTX 5060 Laptop, 8 GB） |
| 优化器 | AdamW（auto） |
| seed | 0 |
| 用时 | 约 227 秒 |

## 最终指标

训练结束后对 `best.pt` 复验（终端 07 截图）：

| class | images | instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 150 | 162 | 0.889 | 0.918 | **0.941** | **0.618** |
| obstacle | 52 | 61 | 0.684 | 0.888 | 0.887 | 0.592 |
| cola | 46 | 47 | 0.988 | 0.979 | 0.975 | 0.736 |
| football | 54 | 54 | 0.995 | 0.889 | 0.961 | 0.527 |

> `results.csv` 最后一轮（epoch 50）记录的是 last.pt 的指标：
> P 0.9168 / R 0.9073 / mAP50 0.9349 / mAP50-95 0.6101；
> 上面表格是训练结束后对 best.pt 的复验结果，报告以 best.pt 为准。

## 关键文件

| 文件 | 说明 |
|---|---|
| `train_20261018_1701/weights/best.pt` | 最佳权重（5,520,209 bytes，约 5.26 MB） |
| `train_20261018_1701/weights/last.pt` | 最后一轮权重 |
| `results.png` | Loss / Precision / Recall / mAP 曲线 |
| `results.csv` | 每个 epoch 的原始指标 |
| `confusion_matrix.png` / `confusion_matrix_normalized.png` | 混淆矩阵 |
| `BoxPR_curve.png` / `BoxF1_curve.png` | PR / F1 曲线 |
| `train_batch*.jpg` | 训练增强后的 batch 可视化 |
| `val_batch*_labels.jpg` / `val_batch*_pred.jpg` | 验证集真实标签 / 预测 |
| `args.yaml` | 训练参数（路径是当时本机的绝对路径，仅作记录） |
| `labels.jpg` | 数据集类别/框分布 |
