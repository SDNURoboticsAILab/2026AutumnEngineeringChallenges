# 基于 YOLO 的机器人场景目标检测 — 项目报告

>
> 仓库：<https://github.com/SDNURoboticsAILab/2026AutumnEngineeringChallenges/tree/main/YOLO>

---

## 目录

1. [项目概述](#1-项目概述)
2. [Level 1 环境配置与数据集整理](#2-level-1-环境配置与数据集整理)
3. [Level 2 目标检测标注](#3-level-2-目标检测标注)
4. [Level 3 模型训练](#4-level-3-模型训练)
5. [Level 4 新图片目标检测](#5-level-4-新图片目标检测)
6. [Level 5 本地图片识别页面](#6-level-5-本地图片识别页面)
7. [遇到的问题与解决方案](#7-遇到的问题与解决方案)
8. [总结与心得](#8-总结与心得)
9. [截图索引与自查清单](#9-截图索引与自查清单)

---

## 1. 项目概述

- **任务**：使用实验室提供的 `obstacle` / `cola` / `football` 三类图片，完成数据整理、目标标注、
  YOLO 模型训练，并用自己训练的模型对训练集外的新图片做目标检测。
- **类别定义**：`0 obstacle`（障碍物）、`1 cola`（可乐）、`2 football`（足球）。
- **技术栈**：Python 3.12 + PyTorch 2.9.1(CPU) + Ultralytics YOLO 8.4.163。
- **本机情况**：Windows，Intel Arc 核显，**无 NVIDIA / CUDA**，因此全流程使用 CPU 训练。

### 完成情况一览

| Level | 内容 | 状态 | 关键结果 |
|---|---|---|---|
| L1 | 环境配置 + 数据集整理 | ✅ | 环境验证通过；数据集 949 张（train 729 / val 186 / test 34），体检 0 损坏 0 重复 |
| L2 | 目标标注 | ✅ | 915 张全部标注完成（手工 219 + 半自动预标注并人工核对 696），1757 个框，自检 0 问题 |
| L3 | 模型训练 | ✅ | YOLO11n（CPU），mAP50 = **0.9443**、mAP50-95 = 0.4705、P = 0.9712、R = 0.8978 |
| L4 | 新图片检测 | ✅ | 对 34 张训练集外新图片完成检测，全部检出、共 74 个目标 |
| L5 | 本地识别页面 | ✅ | Python 标准库实现的本地网页，上传→推理→展示类别与置信度（零额外依赖） |

---

## 2. Level 1 环境配置与数据集整理

### 2.1 环境配置

详细过程见仓库中的 `ENVIRONMENT.md`，要点：

| 组件 | 版本 | 说明 |
|---|---|---|
| 操作系统 | Windows 64 位 | — |
| 显卡 | Intel Arc 核显 | 无 CUDA，选 CPU 版 PyTorch |
| Python | 3.12.10 | 独立虚拟环境 `.venv` |
| PyTorch | 2.9.1+cpu | 与 torchvision 严格配对 |
| torchvision | 0.24.1+cpu | — |
| Ultralytics | 8.4.163 | YOLO11 实现 |

环境验证（截图见 [§9](#9-截图索引与自查清单)）：

```text
python      : 3.12.10
torch       : 2.9.1+cpu
torchvision : 0.24.1+cpu
cuda avail  : False
ultralytics : 8.4.163
```

**截图 1**：Python / PyTorch / Ultralytics 安装成功截图 → `results/01_env_versions.png`

### 2.2 原始数据检查

原始图片来自竞赛仓库 `YOLO/{obstacle,cola,football}`，用 `check_dataset.py` 体检：

| 类别 | 张数 | 格式 | 分辨率 | 备注 |
|---|---|---|---|---|
| obstacle | 316 | .jpg | 640×480（1 张 640×400） | 含 `image_left_*` / `image_rgb_*` 两种前缀 |
| cola | 293 | .jpg | 640×480 | — |
| football | 340 | .jpg | 640×480 | — |
| **合计** | **949** | | | 108.2 MB |

体检发现的问题：**无**（损坏图 0 张、MD5 重复图 0 组、全部图片可正常读取）。
原始数据为**小米杯机器人赛道**实拍：`obstacle` 为赛道上的蓝色方块障碍物，`cola` 为赛道中的可乐罐，
`football` 为球门前的足球；画面中另有气球、机器狗、椅子、球门等非目标物体。

### 2.3 数据集组织与划分

```text
dataset/
├── images/train/   images/val/
├── labels/train/   labels/val/
├── classes.txt
└── split_manifest.csv
```

- **划分方式**：每类先预留 **12 张**（football 因场景组较少为 10 张）**训练集外测试图**
  （`test_images/`，供 Level 4 用），其余按 train:val ≈ 8:2 划分。
- **关键细节（重要）**：原始文件名是连续时间戳采集的（如 `image_rgb_20260727_143017`、
  `143033`、`143121`），**同一采集场次**内相邻帧只差 10~20 秒，画面高度相似；场次之间间隔
  30 分钟以上。如果**纯随机划分**，验证集的图会和训练集高度相似，导致 mAP 虚高、指标不可信。
  因此本项目采用**按采集场次分组**的划分策略：采集间隔 ≤ 30 秒的图片视为同一场次，
  **整个场次**划入 train 或 val，保证验证集来自模型没见过的场景。
  实测分组结果：obstacle 32 组 / cola 13 组 / football 10 组。
- **命名冲突处理**：`obstacle` 目录存在 `image_left_*` 与 `image_rgb_*` 两套前缀，
  统一重命名为 `<类别>_<split>_<序号>.jpg`，避免同名覆盖。

划分结果：

| 类别 | train | val | test（训练集外） |
|---|---|---|---|
| obstacle | 241 | 63 | 12 |
| cola | 225 | 56 | 12 |
| football | 263 | 67 | 10 |
| **合计** | **729** | **186**（20.3%） | **34** |

**截图 2**：整理后的项目目录结构 → `results/02_dataset_tree.png`

---

## 3. Level 2 目标检测标注

### 3.1 标注工具

使用 __________（labelImg / X-AnyLabeling / Roboflow）。

**截图 3**：标注工具运行界面 → `results/03_labelimg.png`

### 3.2 类别编号定义

| 编号 | 类别名 | 含义 |
|---|---|---|
| 0 | obstacle | 障碍物 |
| 1 | cola | 可乐 |
| 2 | football | 足球 |

### 3.3 YOLO 标签格式说明

每张图片对应一个同名 `.txt` 文件，每行 5 个数字：

```text
class_id  cx  cy  w  h
```

- `class_id`：类别编号（0/1/2）；
- `cx, cy`：目标框中心点的 x、y 坐标；
- `w, h`：目标框宽、高；
- 后四个值都**除以图片宽高做归一化**，范围 0~1，因此与图片分辨率无关。

**截图 4**：一张已标注图片 → `results/04_labeled_image.png`
**截图 5**：对应的 YOLO 标签文件内容 → `results/05_label_txt.png`

### 3.4 标注工作量与质量自检

- 标注张数：train **729** 张 + val **186** 张，共 **1757** 个目标框
  （obstacle **803** / cola **345** / football **609**），另有 9 张确认为无目标的背景图。
- 标注方式：**半自动标注（bootstrap 自训练）**——
  ① 先手工标注 219 张（三类各 70~79 张）；
  ② 用这批数据训练一个初版模型（mAP50 0.65 / Recall 0.79）；
  ③ 用初版模型对剩余 472 张自动预标注（脚本 `auto_prelabel.py`，且自动跳过已有手工标签）；
  ④ 人工逐张核对修正（共删除误框 144 个、补画漏框 86 个）。
- 自检脚本 `validate_labels.py` 检查：标签是否一一对应、字段数、类别是否越界、
  坐标是否越界、框是否过小。自检结果：**缺标签 0、孤儿标签 0，问题总数 0 → 可以开始训练**。

---

## 4. Level 3 模型训练

### 4.1 data.yaml

```yaml
path: D:/YOLO/dataset
train: images/train
val: images/val
names:
  0: obstacle
  1: cola
  2: football
```

### 4.2 训练参数

| 参数 | 取值 | 说明 |
|---|---|---|
| 模型 | YOLO11n | 轻量，适合 CPU |
| 预训练权重 | yolo11n.pt | 迁移学习，收敛快、精度高 |
| epochs | 80（设定） | 实际跑到第 20 轮后停止：mAP50 自第 11 轮起稳定在 0.95 附近，已收敛 |
| imgsz | 640 | — |
| batch | 8 | CPU 内存受限 |
| device | cpu | 本机无 CUDA |
| workers | 0 | Windows 下多进程加载数据会报 WinError 5 |
| patience | 20 | 早停 |
| seed | 42 | 结果可复现 |
| 训练数据 | 729 train / 186 val | 手工标注 + 半自动预标注并人工核对 |

启动命令：

```powershell
D:\YOLO\.venv\Scripts\python.exe train.py --epochs 80 --imgsz 640 --batch 8 --name yolo11n_cpu
```

**截图 6**：训练开始时的终端输出 → `results/06_train_start.png`

### 4.3 训练结果

- 结果目录：`runs/train/yolo11n_cpu/`
- **单轮耗时约 130~190 秒**（CPU，729 张图、imgsz 640）
- 说明：因测试机是一台笔记本，中途多次进入睡眠，导致墙钟耗时远超实际计算时间；
  排除睡眠因素后，20 轮的实际计算量约 **50 分钟**。
- 产出：`results.png`（Loss / Precision / Recall / mAP 曲线）、`results.csv`、
  `weights/best.pt`（5.2 MB）、`weights/last.pt`

**截图 7**：训练完成后的结果目录 → `results/07_train_dir.png`
**截图 8**：Loss / Precision / Recall / mAP 曲线（results.png）→ `results/08_results_curve.png`

### 4.4 评估指标

在 186 张验证集图片（328 个目标）上评估：

| 指标 | 数值 |
|---|---|
| Precision | **0.9712** |
| Recall | **0.8978** |
| mAP50 | **0.9443** |
| mAP50-95 | **0.4705** |

各类别指标：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| obstacle | 0.9593 | 0.8640 | 0.9172 | 0.4935 |
| cola | 0.9775 | 0.9649 | 0.9597 | 0.4837 |
| football | 0.9768 | 0.8646 | 0.9559 | 0.4343 |

训练各轮最好成绩（验证集）：mAP50 最高 **0.9544**（第 11 轮），mAP50-95 最高 **0.4717**（第 12 轮）。

**截图 9**：混淆矩阵 confusion_matrix.png → `results/09_confusion_matrix.png`
**截图 10**：权重文件信息（best.pt / last.pt）→ `results/10_weights.png`

---

## 5. Level 4 新图片目标检测

使用**自己训练的** `runs/train/yolo11n_cpu/weights/best.pt`，对 **34 张训练集之外**的新图片检测
（这些图在 `prepare_dataset.py` 划分数据集时就被预留出来，**从未参与训练**）。

```powershell
D:\YOLO\.venv\Scripts\python.exe predict.py --weights runs/train/yolo11n_cpu/weights/best.pt --source test_images
```

结果：**34 张全部检出目标，合计 74 个**，单张目标数分布为
1 个（13 张）/ 2 个（6 张）/ 3 个（11 张）/ 4 个（4 张）。

按 README 要求覆盖三种情况：

| 情况 | 示例图片 | 检测结果 |
|---|---|---|
| ① 单个目标 | `cola_test_003.jpg` | cola 0.87（1 个目标） |
| ② 多个目标 | `obstacle_test_008.jpg` | obstacle 0.92 / obstacle 0.91 / football 0.88 / cola 0.87（4 个目标） |
| ③ 不同场景 / 拍摄角度 | `football_test_004/005/007/008.jpg` | 不同机位下各 3 个目标（obstacle + football 组合） |

检测明细见 `runs/predict/new_images/detections.csv`。

**截图 11**：单目标检测效果 → `results/11_single.png`
**截图 12**：多目标检测效果 → `results/12_multi.png`
**截图 13**：不同场景检测效果 → `results/13_scene.png`

检测明细见 `runs/predict/new_images/detections.csv`。

---

## 6. Level 5 本地图片识别页面

- 实现方式：**Python 标准库 `http.server` + 原生 HTML/JS**（`app_web.py`），
  **不需要安装 Gradio / Streamlit 等任何额外依赖**，离线可用
  （README 允许"其他能够在本地运行的前端或 Web 框架"）
- 页面流程：打开本地页面 → 选择图片 → 后端加载**自己在 Level 3 训练**的 `best.pt`
  → 推理 → 并排展示【原图】与【检测结果图】，并列出目标类别、置信度、边界框

启动命令：

```powershell
D:\YOLO\run_web.cmd
# 或： D:\YOLO\.venv\Scripts\python.exe app_web.py
# 浏览器打开 http://127.0.0.1:7860
```

**截图 14**：页面运行效果 → `results/14_web_ui.png`
（可选）演示视频 → `results/15_demo.mp4`

> 另有一版基于 Gradio 的实现 `app.py`（需先 `pip install gradio`），功能等价；
> 本机为避免额外依赖，实际使用零依赖的 `app_web.py`。

---

## 7. 遇到的问题与解决方案

| # | 问题 | 原因 | 解决方案 |
|---|---|---|---|
| 1 | `ensurepip` 失败，venv 里没有 pip | 受限环境临时目录不可写，无法解包 pip wheel | 直接把 Python 自带的 pip wheel 解包进 site-packages；并把 TEMP 指向项目内 `.tmp` |
| 2 | `from ultralytics import YOLO` 报 `PermissionError: WinError 5` | Ultralytics 默认写 `%APPDATA%\Ultralytics`，该路径不可写 | 在 `sitecustomize.py` 里自动设置 `YOLO_CONFIG_DIR` 到项目内 `.ultralytics` |
| 3 | venv 没有生成 `activate.bat` | 问题 1 导致 ensurepip 提前失败 | `activate_env.cmd` 手动把 `.venv\Scripts` 加入 PATH 并设置 `VIRTUAL_ENV` |
| 4 | 多进程加载数据报 `WinError 5`（命名管道） | 受限环境禁止创建命名管道 | 训练统一用 `workers=0` |
| 5 | 随机划分导致 mAP 虚高 | 连续帧采集，相邻图片高度相似 → 数据泄露 | 改为**按采集场次分组**划分 train/val（间隔 ≤30 秒同组，整组进 val） |
| 6 | `obstacle` 目录图片名冲突 | 同时存在 `image_left_*` 与 `image_rgb_*` 两种前缀 | 统一重命名为 `<类别>_<split>_<序号>.jpg` |
| 7 | labelImg 一滚动/拖到画面边缘就**闪退** | labelImg 1.8.6 的 `scroll_request` 把 float 传给了 `setValue(int)`，新版 PyQt5 直接抛异常并 `qFatal` 退出 | 运行时打补丁改成 `int(...)`（`labelimg_fixes.py`） |
| 8 | 双击 `.cmd` 启动器"黑窗口一闪就没" | 批处理文件里写了中文，cmd.exe 按 GBK 读取 UTF-8 导致解析崩溃 | 所有 `.cmd` 改为**纯 ASCII**，中文提示交给 Python 打印 |
| 9 | 训练"跑了 19 小时才 19 轮" | 笔记本电源设置里"无人参与时系统睡眠超时=5 分钟"，训练被反复冻住（实际计算仅约 50 分钟） | 调整电源计划；并在 `train.py` 里调用 `SetThreadExecutionState` 阻止睡眠 |
| 10 | 自建标注工具预标注会覆盖手工标签 | 预标注脚本对每张图都会写标签文件 | 增加 `--skip-existing` 逻辑：已有非空标签的图默认跳过（`--overwrite` 才覆盖） |
| 11 | 自建网页工具结果图颜色异常 | Ultralytics 对 numpy 输入按 BGR 解释，而 PIL 给的是 RGB | 送进模型前先 `[:, :, ::-1]` 转 BGR，返回时再转回 RGB |
| 12 | （可继续补充） | | |

---

## 8. 总结与心得

- 完整走通了「数据整理 → 标注 → 训练 → 评估 → 推理 → 可视化」的目标检测工程流程；
- 体会到**数据质量比模型结构更重要**：划分方式、标注质量直接决定指标是否可信；
- 在无 GPU 的机器上，通过迁移学习 + 合理的 imgsz/batch 设置，仍然可以完成训练；
- 遇到的问题（临时目录权限、配置目录重定向、多进程管道限制）都通过查文档和看报错逐步定位解决。

---

## 9. 截图索引与自查清单

### 截图索引

| 编号 | 文件 | 内容 |
|---|---|---|
| 1 | `results/01_env_versions.png` | 环境版本验证 |
| 2 | `results/02_dataset_tree.png` | 数据集目录结构 |
| 3 | `results/03_labelimg.png` | 标注工具界面 |
| 4 | `results/04_labeled_image.png` | 已标注图片 |
| 5 | `results/05_label_txt.png` | YOLO 标签文件内容 |
| 6 | `results/06_train_start.png` | 训练启动终端 |
| 7 | `results/07_train_dir.png` | 训练结果目录 |
| 8 | `results/08_results_curve.png` | Loss / P / R / mAP 曲线 |
| 9 | `results/09_confusion_matrix.png` | 混淆矩阵 |
| 10 | `results/10_weights.png` | 权重文件信息 |
| 11~13 | `results/1{1,2,3}_*.png` | Level 4 检测效果（单/多/多场景） |
| 14 | `results/14_web_ui.png` | Level 5 页面运行效果 |

### 逐条自查（对照 README 验证标准）

**Level 1**
- [ ] 提交 Python / PyTorch / Ultralytics 安装成功的终端截图
- [ ] 提交整理后的项目目录结构截图
- [ ] 报告说明训练集与验证集的划分方式

**Level 2**
- [ ] 标注工具运行截图
- [ ] 至少一张已完成目标框标注的图片截图
- [ ] 对应 YOLO 标签文件内容截图
- [ ] 报告说明三个类别的编号及含义

**Level 3**
- [ ] 模型正常开始训练的终端截图
- [ ] 训练完成后的结果目录截图
- [ ] Loss、Precision、Recall 等训练结果图
- [ ] 训练得到的模型权重文件信息
- [ ] 用训练得到的模型完成一次验证或测试

**Level 4**
- [ ] 正确加载自己训练得到的模型
- [ ] 对若干张训练集之外的新图片完成推理
- [ ] 保存目标检测结果
- [ ] 至少一组清晰的最终检测效果图
- [ ] 结果中显示目标框、类别名称、置信度

**Level 5**
- [ ] 本地成功启动前端页面
- [ ] 页面能正常上传图片
- [ ] 调用自己训练的模型完成检测
- [ ] 正确显示结果图、目标类别、置信度
- [ ] 页面运行截图或演示视频
- [ ] 页面源代码及依赖说明

**提交**
- [ ] 完整项目代码（训练、推理、配置文件）
- [ ] 项目报告（本文件）
- [ ] 最终成果展示（训练结果 + 新图片检测）
- [ ] 各 Level 过程记录与截图
- [ ] 以本人学号命名的文件夹 + Pull Request
