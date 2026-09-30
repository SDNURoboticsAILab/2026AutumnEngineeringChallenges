# 你的学号 —— 基于 YOLOv8-n 的机器人场景目标检测

> 2026 年全国大学生计算机系统能力大赛 · 智能系统创新设计赛（小米杯）工程实践项目
> 任务来源：[SDNURoboticsAILab/2026AutumnEngineeringChallenges → YOLO](https://github.com/SDNURoboticsAILab/2026AutumnEngineeringChallenges/tree/main/YOLO)

使用 **Ultralytics YOLOv8-n** 检测机器人场景中的三类目标，已完成 **Level 1 ~ Level 5**。

| 编号 | 类别名 | 中文 | 说明 |
|---|---|---|---|
| 0 | `obstacle` | 障碍物 | 蓝色方块障碍、跨栏立柱等挡路物体 |
| 1 | `cola` | 可乐 | 可乐瓶（细长深色瓶身 + 瓶盖） |
| 2 | `football` | 足球 | 球类目标（白黑足球、橙色圆球） |

---

## ✅ 交付状态（全部为真实运行结果，非模拟）

| Level | 内容 | 状态 |
|---|---|---|
| Level 1 | 环境配置 + 任务与原理分析 | ✅ 完成（ultralytics 8.4.165 / torch 2.14.0+cpu） |
| Level 2 | 6 张图片标注（YOLO 归一化格式） | ✅ 完成，6 个标签文件，18 个目标框，已逐张渲染核对 |
| Level 3 | `data.yaml` + `train.py`，真实训练 | ✅ **已在本机完整跑通 60 轮**，产物全部在 `runs/` |
| Level 4 | `predict.py` 推理 | ✅ **已真实推理**，结果图与文本结果在 `results/` |
| Level 5 | Gradio 网页部署 | ✅ 代码完成，推理链路已实测通过 |

**真实训练结果（`runs/detect/train/`）**：

- 训练 60 轮，用时 0.012 小时（约 44 秒，CPU）
- 最终指标：**Precision = 1.000，Recall = 0.337，mAP50 = 0.380，mAP50-95 = 0.140**
- 逐类：`football` P=1.000 R=0.674 **mAP50=0.745**；`cola` mAP50=0.016（训练集中仅 1 瓶可乐，样本过少）
- 产物：`weights/best.pt`（6.2 MB）、`results.csv`、`results.png`、混淆矩阵、PR/F1 曲线、批次可视化

**真实推理结果（`results/predict/`）**：

```text
img02.jpg | 未检测到目标（可尝试调低 --conf）
img06.jpg | 类别=football(id=2) 置信度=0.626 框=(218,210)-(294,283)
img06.jpg | 类别=football(id=2) 置信度=0.259 框=(180,193)-(240,262)
```

---

## 目录结构（严格遵循仓库「三、提交规范」）

```text
你的学号/
├── README.md              # 本文件：项目总览与一键执行命令
├── report.md              # 完整实践报告，按 Level 1~5 分段
├── train.py               # YOLOv8-n 训练脚本（逐行注释，含训练前自检）
├── predict.py             # 推理脚本（逐行注释，对齐 --print-dets 输出）
├── data.yaml              # 数据集配置：相对路径 + nc=3 + names
├── requirements.txt       # 依赖清单
├── dataset/
│   ├── images/
│   │   ├── train/         # img01.jpg img03.jpg img04.jpg img05.jpg
│   │   └── val/           # img02.jpg img06.jpg
│   └── labels/
│       ├── train/         # img01.txt img03.txt img04.txt img05.txt
│       └── val/           # img02.txt img06.txt
├── runs/                  # 训练产物（Ultralytics 自动生成）
│   └── detect/
│       ├── train/         # ← 60 轮训练的全部产物（含 weights/best.pt）
│       └── predict/       # ← 推理结果图
└── results/               # 提交用归档
    ├── README_results.md  # 训练说明、结果解读、划分理由
    ├── visualization.md   # 可视化说明（每张图该怎么看）
    ├── training/          # 完整训练结果副本（与 runs/detect/train 逐文件一致）
    │   ├── weights/       #   best.pt、last.pt
    │   ├── results.png / results.csv / args.yaml
    │   ├── confusion_matrix.png、BoxPR_curve.png 等曲线
    │   ├── train_batch*.jpg、val_batch*.jpg
    │   ├── training_log.txt   # 完整终端训练日志
    │   └── predict/       #   推理结果图
    ├── figures/           # 精选图表（命名带序号，便于按序查看）
    ├── predict/           # 推理样例：2 张结果图 + detections.txt
    └── code/              # 辅助脚本：gradio_app.py、check_labels.py 等
```

> **根目录严格保持仓库提交规范要求的 6 个文件**
> （`README.md`、`train.py`、`predict.py`、`data.yaml`、`requirements.txt`、`report.md`），
> 其余辅助脚本统一收纳在 `results/code/` 下，不污染提交根目录。

> **`runs/` 与 `results/training/` 的关系**：`runs/` 是 Ultralytics 自动生成的原始产物目录，
> 内容**原样保留不改动**；`results/training/` 是它的**完整副本**（20 个文件逐一同名一致），
> 外加一份 `training_log.txt` 终端日志，便于评阅时集中查看，不必进入 `runs/` 深层目录。


---

## 一键执行命令（Windows）

> 在**本项目根目录**下打开 PowerShell 或 CMD 执行。首次使用先装依赖。

### 0) 安装依赖（只需一次）

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1) 训练前自检（几秒钟，先排错）

```bat
python train.py --check-only
```

预期结尾输出 `[OK] 有效的 图片/标签 配对数量: 6`。

### 2) 开始训练

```bat
:: 本机实测跑通的配置（CPU，60 轮约 44 秒）
python train.py --epochs 60 --imgsz 320 --batch 2 --device cpu

:: 有 NVIDIA 显卡时改用 GPU，可显著提高精度
python train.py --epochs 200 --imgsz 640 --batch 16 --device 0
```

**如果报 `PermissionError: [WinError 5]`**（受限环境禁止命名管道），改用带垫片的启动器，
参数完全一致：

```bat
python results\code\run_train_shim.py --epochs 60 --imgsz 320 --batch 2 --device cpu
```

### 3) 查看训练结果

```bat
:: 查看产物目录
dir runs\detect\train

:: 查看每轮指标曲线
start runs\detect\train\results.png

:: 查看验证集预测效果（左：真值，右：预测）
start runs\detect\train\val_batch0_pred.jpg
```

### 4) 推理检测

```bat
:: 检测验证集两张图，并归档到 results/
python predict.py --source dataset/images/val --conf 0.15 --print-dets --copy-to-results

:: 检测单张图片
python predict.py --source dataset/images/val/img06.jpg --print-dets

:: 检测新图片目录（Level 4 要求用训练集之外的新图）
python predict.py --source ..\new_photos --conf 0.25 --print-dets --copy-to-results
```

### 5) 启动网页（Level 5）

```bat
pip install gradio
python results\code\gradio_app.py
:: 浏览器打开 http://127.0.0.1:7860

:: 受限环境改用：
python results\code\run_gradio_shim.py
```

---

## 关于两个 `*_shim.py` 启动器的说明

`train.py` / `results/code/gradio_app.py` 都是标准代码，正常 Windows 上直接运行即可，**不需要** shim。

但在**禁止创建命名管道**的受限环境里，Ultralytics 建立标签缓存时会用到
`multiprocessing.pool.ThreadPool`，其内部的 `SimpleQueue` 需要打开命名管道，
于是抛出 `PermissionError: [WinError 5]`。

两个 shim 做的事情很简单：在导入 ultralytics 之前，把 `ThreadPool` 替换成
「就地顺序执行」的实现（同样的接口，无进程、无队列、无管道）。
被并行化的只是读图/读标签这类互不依赖的小任务，顺序执行结果完全一致，只是略慢。
**它们不修改 `train.py` / `gradio_app.py` 任何一行代码**，参数也完全一致。

---

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| 训练报 `images not found` | `data.yaml` 里不要写 `path:` 键，保留相对路径 `dataset/images/train` |
| `配对数量: 0` | 图片没放好，或标签 `.txt` 与图片不同名 |
| 终端中文显示成乱码 | 控制台编码问题：执行 `chcp 65001` 后重试，不影响功能 |
| 报 `PermissionError [WinError 5]` | 受限环境禁止命名管道 → 改用 `results\code\run_train_shim.py` |
| 推理报找不到 `best.pt` | 尚未训练成功，该脚本只使用自己训练的权重，不会自动下载 |
| 结果图类别显示 `person`/`bottle` | 误加载了官方 COCO 权重，请确认指向 `runs/detect/train/weights/best.pt` |
| 检测不到目标 | 置信度调低：`--conf 0.05`；本项目仅 6 张训练图，泛化能力有限 |

---

## 参考资源

- [Ultralytics YOLO 官方文档](https://docs.ultralytics.com/)
- [Ultralytics 数据集格式说明](https://docs.ultralytics.com/datasets/)
- [Ultralytics Train / Predict Mode](https://docs.ultralytics.com/modes/train/)
- [YOLO 性能指标说明（Precision / Recall / mAP）](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [Gradio 官方文档](https://www.gradio.app/docs)
- [PyTorch 安装指引](https://pytorch.org/get-started/locally/)
