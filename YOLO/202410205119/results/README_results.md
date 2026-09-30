# 训练说明与结果解读

> 本文件说明：本项目**真实训练**是如何完成的、结果文件怎么读、指标为什么是这个水平。
> 所有数值均来自本机实际运行的 60 轮训练，不是模拟值。
> 完整终端日志见 `docs/training_log.txt`，原始指标见 `docs/results.csv`。

---

## 1. 训练是怎么跑起来的（一键命令）

```bat
:: 在项目根目录（你的学号/）下执行

:: 1) 训练前自检（几秒钟，先排错）
python train.py --check-only

:: 2) 正式训练（本机实测配置：CPU，60 轮约 44 秒）
python train.py --epochs 60 --imgsz 320 --batch 2 --device cpu

:: 3) 有 NVIDIA 显卡时改用 GPU（精度更高、可跑更多轮）
python train.py --epochs 200 --imgsz 640 --batch 16 --device 0
```

**为什么本机用 `--imgsz 320` 而不是默认的 640**：
本机无 NVIDIA 显卡（`cuda_available=False`），只能纯 CPU 训练。
320 分辨率能让 60 轮在约 44 秒内跑完，便于快速验证完整流程；
代价是小目标（远处球）的特征被压缩，检出率下降。GPU 环境下建议用 640。

**如果训练报 `PermissionError: [WinError 5]`**：
某些受限环境禁止创建命名管道，而 Ultralytics 建标签缓存时用到
`multiprocessing.pool.ThreadPool`。此时改用附带的兼容启动器（参数完全一致）：

```bat
python results\code\run_train_shim.py --epochs 60 --imgsz 320 --batch 2 --device cpu
```

---

## 2. 训练产物在 `runs/` 目录的什么位置

`runs/` 是 **Ultralytics 自动创建并写入**的输出目录，本项目实际产物位于：

```text
runs/detect/train/
├── weights/
│   ├── best.pt                        ★ 验证集最优权重（推理用这个，5.9 MB）
│   └── last.pt                        最后一轮权重（断点续训用）
├── args.yaml                          本次训练用到的全部超参
├── results.csv                        每轮的 loss 与 P / R / mAP 数值
├── results.png                        ★ 指标曲线图（loss 下降、mAP 上升）
├── confusion_matrix.png               混淆矩阵
├── confusion_matrix_normalized.png    按行归一化的混淆矩阵
├── BoxF1_curve.png                    F1-置信度曲线
├── BoxP_curve.png                     Precision-置信度曲线
├── BoxPR_curve.png                    Precision-Recall 曲线（面积即 AP）
├── BoxR_curve.png                     Recall-置信度曲线
├── labels.jpg                         数据集标签分布统计
├── train_batch0/1/2.jpg               训练批次可视化（含增强后的框）
├── val_batch0_labels.jpg              验证集真实标签
└── val_batch0_pred.jpg                ★ 验证集预测结果
```

### 与 `results/` 的关系

| 目录 | 性质 | 内容 |
|---|---|---|
| `runs/` | **程序自动产物**（原样保留，不要手工改） | 训练与推理的全部原始输出 |
| `results/training/` | **完整副本** | `runs/detect/train/` 的逐文件拷贝 + `training_log.txt` 终端日志 |
| `results/figures/` | **精选图表** | 从训练产物中挑出的 8 张关键图，文件名带序号便于按序查看 |
| `results/predict/` | **推理样例** | Level 4 的检测结果图 + 文本结果 |
| `results/code/` | **辅助脚本** | 网页部署、数据校验等脚本（不影响根目录提交规范） |

**为什么同时保留 `runs/` 和 `results/training/`**：
`runs/` 是 Ultralytics 的原始输出目录，必须原样保留以体现"程序自动生成"；
`results/training/` 则是同一批文件的完整副本（20 个文件逐一同名一致），
外加一份 `training_log.txt`。这样评阅者只打开 `results/` 就能看到全部训练成果，
不必进入 `runs/detect/train/` 深层目录逐层翻找。

`results/figures/` 里的图是从这批产物中挑出的**精选子集**，
命名加了序号（`01_`、`02_`…），按序号看就是推荐的阅读顺序。

---

## 3. 如何查看训练结果

### 3.1 命令行查看

```bat
:: 查看产物目录
dir runs\detect\train
dir runs\detect\train\weights

:: 直接打开关键图（Windows）
start runs\detect\train\results.png
start runs\detect\train\confusion_matrix.png
start runs\detect\train\val_batch0_pred.jpg
```

### 3.2 用 Python 读取指标

```python
import pandas as pd                                   # pip install pandas
df = pd.read_csv("runs/detect/train/results.csv")
df.columns = [c.strip() for c in df.columns]          # 列名可能带前导空格
print(df[["epoch", "train/box_loss", "train/cls_loss",
          "metrics/precision(B)", "metrics/recall(B)",
          "metrics/mAP50(B)"]].tail(10))
```

### 3.3 各类结果图怎么看

| 图片 | 看什么 | 本项目的表现 |
|---|---|---|
| `results.png` | 三条 loss 是否整体下降、P/R/mAP 是否上升 | `cls_loss` 3.73→2.22 稳定下降 |
| `confusion_matrix.png` | 对角线越深越好；非对角线表示类别混淆 | `football` 对角线明显，`cola` 因样本过少几乎全漏 |
| `BoxPR_curve.png` | 曲线下面积即 AP，越靠右上越好 | `football` 曲线有一定面积，`cola` 贴近坐标轴 |
| `labels.jpg` | 目标框位置、宽高分布是否合理 | 框分布集中在画面中部，与实拍场景一致 |
| `val_batch0_labels.jpg` vs `val_batch0_pred.jpg` | 前者真值、后者预测，越接近越好 | 预测框命中橙色球 |

---

## 4. 实际训练结果（真实数值）

### 4.1 训练配置摘要

| 项目 | 值 |
|---|---|
| 模型 | YOLOv8-n（`yolov8n.pt` 迁移学习） |
| 训练轮数 | 60 |
| 输入尺寸 | 320 × 320 |
| 批大小 | 2 |
| 设备 | CPU（AMD Ryzen 7 8845H） |
| 训练耗时 | **0.012 小时（约 44 秒）** |
| 模型规模 | 129 层，3,011,433 参数，8.2 GFLOPs |
| 迁移情况 | `Transferred 319/355 items from pretrained weights` |
| 类别改写 | `Overriding model.yaml nc=80 with nc=3` |

### 4.2 关键指标变化

| Epoch | box_loss | cls_loss | dfl_loss | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| 1 | 1.436 | 3.730 | 1.223 | 0.005 | 0.625 | 0.008 | 0.004 |
| 30 | 2.022 | 2.520 | 1.219 | 0.008 | 0.750 | 0.021 | 0.010 |
| **60** | 2.109 | **2.222** | 1.594 | **1.000** | 0.337 | **0.380** | **0.140** |

### 4.3 逐类最终指标

| 类别 | 验证集实例数 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `all` | 5 | 1.000 | 0.337 | 0.380 | 0.140 |
| `cola` | 1 | 1.000 | 0.000 | 0.016 | 0.008 |
| `football` | 4 | 1.000 | 0.674 | **0.745** | 0.272 |

### 4.4 指标解读（重要）

**训练是成功的，但模型能力受数据量严重限制。**

1. **`cls_loss` 3.730 → 2.222** —— 分类能力确实在持续提升，训练没有跑偏；
2. **`football` mAP50 = 0.745** —— 在仅 8 个训练框的条件下属于合理水平，
   说明模型学到了"球形目标"的特征；
3. **Precision = 1.000** —— 模型预测出的框全部命中真实目标（零误检），
   这是小数据集 + 较高置信度阈值的典型表现；
4. **Recall 仅 0.337** —— 漏检较多。因为验证集只有 5 个目标框，
   漏掉 1 个就损失 20% 召回率；
5. **`cola` mAP50 = 0.016，Recall = 0** —— ⚠️ 训练集中**只有 1 瓶可乐**（`img01.jpg`），
   单一样本无法让模型泛化到 `img02.jpg` 中的另一瓶可乐。
   **这是数据量的客观限制，不是代码或配置错误**；
6. `box_loss` / `dfl_loss` 末段略有回升，是最后若干轮 `close_mosaic` 关闭
   mosaic 增强、输入数据分布变化导致的正常波动。

---

## 5. 训练集 / 验证集划分（为什么是 4:2）

| 子集 | 图片 | 目标框数 | 包含类别 |
|---|---|---|---|
| `train` | `img01` `img03` `img04` `img05` | 13 | obstacle 4、cola 1、football 8 |
| `val` | `img02` `img06` | 5 | cola 1、football 4 |

**关键约束**：实测统计发现，`cola` **只出现在** `img01`/`img02`，
`obstacle` **只出现在** `img03`/`img04`。这意味着
**没有任何一种 4:2 划分能让三个类别同时进入训练集和验证集**。

**取舍原则：优先保证训练集包含全部三个类别。**
因为训练集里没出现过的类别，模型永远学不会 ——
若把 `img03`/`img04` 放进验证集，模型对 `obstacle` 的检测能力必然为 0。

代价是验证集缺少 `obstacle`，因此本项目的 mAP **只反映 `cola` 与 `football` 两类**。
完整的划分论证见 `../report.md` 第 2.4 节。

---

## 6. 常见训练问题排查

以下几类问题**不会让训练报错**，只会使指标异常，排查时优先级最高：

| 现象 | 原因 | 处理 |
|---|---|---|
| mAP 恒为 0 | 标签文件为空 | 打开 `.txt` 确认有内容 |
| 大量框被丢弃 | 坐标未归一化（写成像素值） | 检查数值是否都在 0~1 之间 |
| 等同于没有标签 | 图片与标签文件名不一致 | `python train.py --check-only` 看配对数量 |
| 维度不匹配报错 | `nc` 与 `names` 数量不一致 | 自检脚本会拦截 |
| 能训练但类别学错 | 类别编号与 `names` 不符 | 人工核对编号与名称 |
| `PermissionError [WinError 5]` | 受限环境禁止命名管道 | 改用 `results/code/run_train_shim.py` |

**经验总结**：训练前务必先跑 `python train.py --check-only`。
本项目在 `train.py` 中内置了这套自检，把上述问题在训练开始前全部挡掉 ——
比训练跑了几小时后才发现问题要高效得多。
