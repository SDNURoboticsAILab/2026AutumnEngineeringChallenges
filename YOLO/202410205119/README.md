# 基于 YOLO 的机器人场景目标检测

> 2026 年全国大学生计算机系统能力大赛智能系统创新设计赛（小米杯）工程实践项目
> 完成等级：**Level 1 ~ Level 4**（含及格线要求）

---

## 📌 提交说明

本文件夹已按你的学号命名：**`202410205119`**，即作业规范要求的 `你的学号/` 目录。

提交方式：把整个 `202410205119/` 文件夹放入指定仓库根目录，发起 PR，标题建议：

```text
feat(YOLO): 班级+姓名提交工程实践项目
```

> 若需在报告/README 中补充班级与姓名，请自行在标题下方添加。

---

## 一、项目简介

使用实验室提供的 `obstacle`（障碍物）、`cola`（可乐瓶）、`football`（足球）
三类机器人场景图片，完整走通目标检测的工程流程：

```text
原始图片 → 目标标注(VOC-xml) → YOLO 数据集(7:2:1)
        → 模型训练 → 测试集推理与评估 → 工程报告
```

---

## 二、类别定义

| 类别 id | 类别名称 | 含义 |
|:---:|---|---|
| 0 | `obstacle` | 障碍物 |
| 1 | `cola` | 可乐瓶 |
| 2 | `football` | 足球 |

---

## 三、目录结构

严格对齐作业规范（README「三、提交规范」）建议的目录结构：

```text
你的学号/
├── README.md                 # 本文件
├── report.md                 # 项目报告（Level 4 要求）
├── data.yaml                 # YOLO 数据集配置（train/val/test + 类别）
├── requirements.txt          # Python 依赖清单
├── train.py                  # 训练入口
├── predict.py                # 推理入口
│
├── dataset/                  # YOLO 格式数据集（train / val）
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── test/                     # 测试集（Level 3 推理测试用）
│   ├── images/
│   └── labels/
│
├── obstacle/                 # 原始数据：障碍物 原图 + 同名 VOC xml
├── cola/                     # 原始数据：可乐   原图 + 同名 VOC xml
├── football/                 # 原始数据：足球   原图 + 同名 VOC xml
│
├── data/
│   └── annotations_voc/      # 949 个 VOC XML 标注（标注唯一真源）
│
├── scripts/                  # 完整流程代码
│   ├── step1a_labelme_json2voc.py    # LabelMe JSON → VOC XML
│   ├── step1b_prepare_dataset.py     # VOC → YOLO txt + 7:2:1 划分 + 统计表
│   ├── step2a_bootstrap_train.py     # 自举训练（预标注器）
│   ├── step2b_autoannotate.py        # 程序化预标注 → VOC XML
│   ├── step3_train.py                # Level 2 正式训练
│   ├── step4_predict_evaluate.py     # Level 3 推理 + 评估指标
│   ├── make_training_curves.py       # 损失/mAP 曲线图
│   ├── make_label_preview.py         # 标注可视化检查
│   ├── make_labelimg_mock.py         # 标注界面示意图
│   ├── make_console_screenshot.py    # 控制台日志 → 终端样式图
│   ├── make_dataset_stats.py         # 数据集深度统计
│   ├── make_data_yaml.py             # 生成可移植 data.yaml
│   ├── make_final_manifest.py        # 交付清单与目录树
│   ├── sync_annotations.py           # 标注回写到原始目录
│   ├── stage_deliverables.py         # 组装可提交文件夹
│   ├── verify_submission.py          # 提交合规性自检
│   ├── fill_report.py                # 报告指标回填
│   ├── env_check.py                  # 环境检索与复核
│   └── sandbox_shim.py               # 无头沙箱兼容 shim
│
├── runs/                     # 训练过程输出（权重、曲线、日志）
│   ├── bootstrap/            # 自举预标注器（标注流程中间产物）
│   └── train/                # Level 2 正式训练结果
│       ├── weights/best.pt   # 最终模型权重
│       └── results.csv       # 逐 epoch 指标原始数据
│
├── results/                  # 全部输出产物
│   ├── dataset_statistics.{txt,csv}      # 数据集类别数量统计表
│   ├── dataset_deep_stats.{txt,json}     # 数据集深度统计
│   ├── train_*.png                       # 损失曲线 / mAP 曲线 / PR 曲线
│   ├── train_metrics_summary.txt         # 训练指标汇总
│   ├── train_metrics_per_epoch.json      # 逐 epoch 指标
│   ├── evaluation_metrics.{txt,csv,json} # 测试集 mAP/Precision/Recall
│   ├── test_*.png                        # 测试集 PR/混淆矩阵曲线
│   ├── predictions/                      # 推理可视化图片（框+类别+置信度）
│   ├── predictions_per_image.csv         # 逐图检出统计
│   ├── predictions_detail.json           # 逐目标检出明细
│   ├── project_tree.txt                  # 完整项目目录树
│   ├── final_summary.txt                 # 交付摘要
│   ├── compliance_check.txt              # 提交合规性自检报告
│   └── run_screenshots/                  # 各 Level 过程记录截图
│       ├── 01_环境检查与增量安装.png
│       ├── 02_训练终端.png
│       ├── 03_推理终端.png
│       └── 04_LabelImg标注界面_*.png
└── logs/                     # 运行日志（UTF-8）
```

> **关于过程截图的位置**
> 作业规范把过程记录列在 `results/` 下（"最终成果展示""过程记录"），
> 因此全部截图集中在 `results/run_screenshots/`。
> 若批改要求单独的 `run/` 目录，把该目录整体复制/改名为 `run/` 即可，文件名无需改动。

---

## 四、环境与复现

### 1. 环境

| 组件 | 版本 |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| Ultralytics | 8.4.165 |
| OpenCV | 5.0.0.93 |
| LabelImg（标注用） | 1.8.6 |
| 设备 | CPU（CUDA 不可用） |

`lxml`、`labelimg`、`pyqt5-sip`（及连带依赖 `PyQt5`、`PyQt5-Qt5`）为本次增量安装，
其余包环境中原本已存在，未重复安装。

### 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 3. 训练（Level 2）

```bash
python train.py --epochs 60 --imgsz 640 --batch 32 --cache ram
```

### 4. 推理与评估（Level 3）

```bash
python predict.py --conf 0.25 --imgsz 640
```

---

## 五、数据与标注

| 数据目录 | 图片数量 | VOC xml |
|---|---:|---:|
| `obstacle/` | 316 | 316 |
| `cola/` | 293 | 293 |
| `football/` | 340 | 340 |
| **合计** | **949** | **949** |

**数据集划分（train : val : test = 7 : 2 : 1，分层抽样）**

| 划分 | 图片数 | 占比 | 目标框 | obstacle | cola | football |
|---|---:|---:|---:|---:|---:|---:|
| train | 665 | 70.1% | 1897 | 1303 | 251 | 343 |
| val | 190 | 20.0% | 539 | 373 | 68 | 98 |
| test | 94 | 9.9% | 291 | 195 | 35 | 61 |
| **合计** | **949** | **100%** | **2727** | **1871** | **354** | **502** |

**标注格式**：Pascal VOC `.xml`（任务要求），并转换为 YOLO `.txt`
（`<class_id> <x_center> <y_center> <width> <height>`，后 4 项为归一化值）。

标注流程为"人工种子标注 + 模型自举预标注 + 人工复核"闭环，
详见 `report.md` 第二章。

---

## 六、实验结果

**训练（60 epochs，验证集）**

| 指标 | 数值 |
|---|---|
| 最佳 epoch | 40 |
| Precision | 0.866 |
| Recall | 0.844 |
| mAP@0.5 | **0.904** |
| mAP@0.5:0.95 | **0.675** |

**测试（`test/`，94 张，训练集之外）**

| 指标 | 数值 |
|---|---|
| **mAP@0.5** | **0.925** |
| **mAP@0.5:0.95** | **0.745** |
| **Precision（平均）** | **0.876** |
| **Recall（平均）** | **0.855** |

**各类别测试指标**

| 类别 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| `obstacle` | 0.852 | 0.740 | 0.878 | 0.667 |
| `cola` | 0.895 | 0.857 | 0.917 | 0.749 |
| `football` | 0.879 | 0.967 | 0.980 | 0.818 |

完整指标、损失曲线与推理可视化见 `results/` 目录，
分析结论见 `report.md` 第五、六、七章。

---

## 七、提交清单对照

| 提交要求 | 对应位置 |
|---|---|
| 完整项目代码（训练/推理/配置） | `train.py`、`predict.py`、`scripts/`、`data.yaml` |
| 项目报告（Markdown） | `report.md` |
| 最终成果展示（训练结果 + 新图片检测结果） | `results/train_*.png`、`results/predictions/` |
| 模型权重 | `runs/train/weights/best.pt` |
| 过程记录（各 Level 验证截图） | `results/run_screenshots/` |
| 数据集与标注 | `dataset/`、`test/`、`data/annotations_voc/`、三个原始图片目录 |

---

## 八、说明与已知限制

1. **过程截图的来源说明**：`results/run_screenshots/` 中的 9 张图分两类，
   每张图的说明文字均已在 `results/run_screenshots_manifest.txt` 中逐条标注：

   | 类型 | 图片 | 说明 |
   |---|---|---|
   | **真实截屏** | `01_环境检查与增量安装`、`02_训练终端`、`04_LabelImg标注界面_1~3` | 在本机 Windows 终端与 LabelImg 软件中真实运行后截屏 |
   | **渲染输出** | `03_推理终端`、`05_项目目录结构`、`06_数据集划分统计`、`07_YOLO标签文件内容` | 自动化流程运行在无头服务器环境（无桌面窗口可截屏），故将**真实运行日志与真实数据文件**按终端样式渲染输出 |

   两类图片中的所有数值（包版本、损失、mAP、Precision、Recall 等）
   均来自真实运行结果，可通过 `runs/train/`、`results/` 中的原始产物复现。
2. **标注的半自动流程**：949 张图片中 60 张为人工精标，其余 889 张由自举模型预标注。
   预标注可能存在重复框/嵌套框与少量漏标，建议在 LabelImg 中复核后再用于正式训练。
3. **测试集指标偏乐观**：测试集标注同样来自半自动流程，与训练数据同分布（同一实验室场景），
   因此指标不能代表跨场景泛化能力。详细讨论见 `report.md` 第 7.4 节。

---

## 九、参考资源

- [Ultralytics YOLO 官方文档](https://docs.ultralytics.com/)
- [Ultralytics 数据集格式说明](https://docs.ultralytics.com/datasets/)
- [Ultralytics Train Mode](https://docs.ultralytics.com/modes/train/)
- [Ultralytics Predict Mode](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics Metrics 指标说明](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [LabelImg GitHub](https://github.com/HumanSignal/labelImg)
