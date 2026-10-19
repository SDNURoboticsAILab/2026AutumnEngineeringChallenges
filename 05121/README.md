# 基于 YOLO 的机器人场景目标检测 — 提交说明

**学号**：202410205121　**姓名**：刘欢＿＿＿＿＿　**班级**：＿＿＿＿＿＿

---

## 一、项目简介

使用实验室提供的 `obstacle`（障碍物方块）、`cola`（可乐）、`football`（足球）三类图片，
完成**数据整理 → 目标标注 → YOLO 模型训练 → 新图片检测 → 本地识别页面**的完整流程。
全流程在无 NVIDIA 显卡的 Windows 笔记本上使用 **CPU** 完成。

**最终结果（验证集 186 张 / 328 个目标）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| obstacle | 0.9593 | 0.8640 | 0.9172 | 0.4935 |
| cola | 0.9775 | 0.9649 | 0.9597 | 0.4837 |
| football | 0.9768 | 0.8646 | 0.9559 | 0.4343 |
| **总体** | **0.9712** | **0.8978** | **0.9443** | **0.4705** |

---

## 二、目录结构

```text
202410205121/
├── README.md               本文件
├── report.md               项目报告（环境/数据/标注/训练/推理/页面/问题与解决）
├── ENVIRONMENT.md          环境配置详细说明（版本、安装步骤、踩坑记录）
├── data.yaml               数据集配置（Ultralytics 格式）
├── requirements.txt        依赖清单
├── train.py                训练脚本（CPU 参数已调好，含防睡眠处理）
├── predict.py              推理脚本（对新图片检测，输出结果图 + detections.csv）
├── app_web.py              Level 5 本地识别页面（Python 标准库实现，零额外依赖）
├── tools/                  开发过程中用到的辅助脚本
│   ├── check_dataset.py        数据集体检（张数/分辨率/损坏/重复）
│   ├── prepare_dataset.py      划分 train/val、预留测试图、重命名、生成 data.yaml
│   ├── validate_labels.py      标签自检（漏标/越界/类别错误）
│   ├── auto_prelabel.py        半自动预标注（用初版模型给剩余图片打框）
│   ├── make_subset.py          生成只含已标注图片的子集（bootstrap 用）
│   └── label_tool.py           自建的网页标注工具（点两下画框）
├── dataset/
│   ├── classes.txt             类别清单（0 obstacle / 1 cola / 2 football）
│   ├── split_manifest.csv      划分清单（每张图来自哪个原始文件）
│   └── labels/{train,val}/     YOLO 格式标签（共 1757 个目标框）
├── runs/
│   ├── train/yolo11n_cpu/      训练结果：results.png（曲线）、results.csv、args.yaml、
│   │                           train_batch*.jpg、weights/best.pt（最终权重）
│   └── val/final_eval/         验证结果：混淆矩阵、PR/P/R/F1 曲线、val_batch*_pred.jpg
└── results/                    报告用截图（按 report.md 第 9 节的编号命名）
```

> **数据集说明**：原始图片（949 张，约 108 MB）体积较大，未随包提交。
> 图片来自竞赛仓库 <https://github.com/SDNURoboticsAILab/2026AutumnEngineeringChallenges/tree/main/YOLO>
> 的 `obstacle` / `cola` / `football` 三个目录。
> 本包内的 `dataset/labels/` 已包含全部标注（很小），配合原始图片即可完全复现训练。

---

## 三、如何运行

### 1. 环境

```powershell
# 方式 A：一键脚本（需要先有 Python 3.12）
powershell -ExecutionPolicy Bypass -File setup_env.ps1    # 见 ENVIRONMENT.md

# 方式 B：手动
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu ^
    --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 训练

```powershell
.\.venv\Scripts\python.exe train.py --data data.yaml --epochs 80 --imgsz 640 --batch 8 --name yolo11n_cpu
```

### 3. 新图片检测（Level 4）

```powershell
.\.venv\Scripts\python.exe predict.py --weights runs\train\yolo11n_cpu\weights\best.pt --source test_images
```
> `test_images/` 是划分数据集时预留的 34 张训练集外图片（本包未含，可从原始数据按
> `dataset/split_manifest.csv` 中 `split=test` 的记录还原）。

### 4. 本地识别页面（Level 5）

```powershell
.\.venv\Scripts\python.exe app_web.py
# 浏览器打开 http://127.0.0.1:7860，选择图片即可看到检测结果
```

---

## 四、实现要点

1. **按采集场次划分数据集**：原始图片是连续时间戳采集的，相邻帧画面高度相似。
   若随机划分 train/val 会造成数据泄露、指标虚高。本项目把采集间隔 ≤30 秒的图片归为同一场次、
   整场次划入 val，保证验证集来自模型没见过的场景。
2. **半自动标注（bootstrap）**：先手工标注 219 张 → 训练初版模型 → 用它对剩余 472 张
   自动预标注 → 人工逐张核对修正。相比纯手工标注 915 张，效率显著提升。
3. **CPU 训练调优**：无 CUDA 环境下使用迁移学习（yolo11n.pt）、`workers=0`（规避 Windows
   多进程加载数据报错）、并调用 `SetThreadExecutionState` 防止笔记本睡眠中断训练。
4. **零依赖的 Level 5 页面**：用 Python 标准库 `http.server` + 原生 HTML/JS 实现，
   不需要安装 Gradio / Streamlit。

---

## 五、报告与截图

- 完整报告见 `report.md`
- 截图见 `results/` 目录，编号与 `report.md` 第 9 节的"截图索引"一一对应
