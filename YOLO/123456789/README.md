# 基于 YOLO 的机器人场景目标检测 —— 项目提交说明

> **学生**：C　|　**学号**：123456789　|　**提交日期**：2026-09-28
> **提交目录**：`YOLO/123456789/`
> **考核项目**：SDNU 机器人实验室 2026 秋季工程挑战 · 项目一

---

## 一、项目概况

使用实验室提供的 `obstacle`（障碍物）、`cola`（可乐瓶）、`football`（足球）三类图片，
完成「数据整理 → 目标标注 → YOLO 模型训练 → 新图片目标检测」的完整工程链路。

**当前完成情况**：

| Level | 内容 | 状态 |
| --- | --- | --- |
| Level 1 | 准备 YOLO 开发环境与整理数据集 | ✅ 已完成 |
| Level 2 | 完成目标检测标注 | ⚠️ **进行中**（见 4.4 节：obstacle 类待补 209 张） |
| Level 3 | 完成 YOLO 模型训练 | ✅ 已完成 |
| Level 4 | 完成新图片目标检测（**及格线**） | ✅ **已完成**（5/5 验收标准通过） |
| Level 5 | 制作本地图片识别页面（进阶） | ⬜ 未开始 |

---

## 二、环境配置（Level 1 第 1 节）

| 项目 | 版本 / 说明 |
| --- | --- |
| 操作系统 | Windows |
| GPU | NVIDIA GeForce RTX 3050 Laptop（4 GB 显存） |
| Python | 3.14.7 |
| PyTorch | 2.14.0+cu126（CUDA 12.6 构建） |
| TorchVision | 0.29.0+cu126 |
| Ultralytics | 8.4.155 |
| OpenCV | 5.0.0 |
| 标注工具 | LabelImg 1.8.6 |

**虚拟环境位置**：`C:\Users\dell\yolo-env`（刻意使用 ASCII 路径，避开中文与空格路径引发的异常）。

```powershell
# 安装（可重复执行，已装好会自动跳过）
.\setup_env.ps1

# 验证：应打印版本号、True 与显卡型号
C:\Users\dell\yolo-env\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> `requirements.txt` 记录了全部依赖版本。
> 提交目录**不包含** `.venv/`（本地虚拟环境，已在打包时排除，避免把上万个依赖文件提交进仓库）。

---

## 三、数据集（Level 1 第 2、3 节）

### 3.1 原始数据

共 **949 张** 640×480 图片，按来源目录分为三类：

| 来源目录 | 数量 |
| --- | --- |
| obstacle | 316 |
| cola | 293 |
| football | 340 |

全部图片均可正常打开、格式统一，检查脚本为 `check_rawdata.py`。

### 3.2 划分方式

按**类别分层随机划分**，训练集 : 验证集 = **8 : 2**，随机种子 `seed=42`，脚本为 `split_dataset.py`。

> 说明：项目早期曾用 PowerShell（`split_dataset.ps1`）做过一次划分，与 Python 版随机过程不同，
> 会导致同一批图片落到不同 split。最终统一以 `split_dataset.py` 为唯一划分入口并重新生成数据集，
> 已标好的 219 个标签按图片实际所在 split 重新归位，做到"换划分不丢标注"。

### 3.3 目录结构

```text
dataset/
├── images/
│   ├── train/      755 张训练图片
│   └── val/        188 张验证图片
├── labels/
│   ├── train/      676 个标签文件 + classes.txt
│   └── val/        171 个标签文件 + classes.txt
└── _holdout_backup_*/   Level 4 新图片移出前的备份（可回滚）
```

YOLO 按**同名文件**配对：`dataset/images/train/xxx.jpg` ↔ `dataset/labels/train/xxx.txt`。

### 3.4 当前标注规模

| 划分 | 图片 | 有标签 | 无标签（背景图） | obstacle | cola | football |
| --- | --- | --- | --- | --- | --- | --- |
| train | 755 | 676 | 79 | 94 | 253 | 623 |
| val | 188 | 171 | 17 | 27 | 58 | 154 |
| **合计** | **943** | **847** | **96** | **121** | **311** | **777** |

标注框合计 **1209 个**。另有 6 张图片已主动移出数据集，作为 Level 4 的"新图片"。

---

## 四、目标标注（Level 2）

### 4.1 类别定义

| 编号 | 类别 | 含义 |
| --- | --- | --- |
| 0 | obstacle | 障碍物 |
| 1 | cola | 可乐瓶 |
| 2 | football | 足球 |

顺序定义在 `classes.txt`，必须与 `data.yaml` 的 `names`、标签文件每行第一个数字**三者一致**。

### 4.2 标注工具

**主工具：LabelImg 1.8.6**（考核推荐工具，本地离线运行，原生输出 YOLO txt）。

- 两套环境均可使用：`C:\Users\dell\yolo-env`（Python 3.14.7 + PyQt5 5.15.11）、
  `C:\Users\dell\label-env`（Python 3.11.9 + PyQt5 5.15.9）
- **必须打补丁才能稳定运行**：labelImg 1.8.6 与新版 PyQt5 不兼容，会出现"按 `W` 画框即闪退"
  （退出码 `0xC0000409`，终端无任何报错）。共 6 个幂等补丁脚本：
  `_patch_labelimg.py`（`setValue` 的 float→int）、`_fix_labelimg_painter.py`
  （`QPainter.drawRect/drawLine/drawText` 的 float→int）、`_fix_labelimg_w_key.py`（快捷键冲突）、
  `_fix_labelimg_asserts.py`、`_fix_libs_asserts.py`（热路径 assert 改优雅处理）、
  `_fix_classes_txt.py`（补齐 YoloReader 需要的 classes.txt）
- 实测 PyQt5 5.15.6 / 5.15.9 + sip 12.11~12.13 同样拒收 float，5.15.2 / 5.15.4 无对应轮子，
  即"免补丁即可用"的版本组合不存在

**启动方式**（第三个参数 `save_dir` 让标注结果直接写进 `dataset/labels/<split>/`，无需事后搬运）：

```powershell
C:\Users\dell\label-env\Scripts\labelImg.exe "D:\研一\git 练习\yolo_project\dataset\images\train" "D:\研一\git 练习\yolo_project\classes.txt" "D:\研一\git 练习\yolo_project\dataset\labels\train"
```

项目内已封装双击启动脚本：`7_/8_`（3.14 环境）、`10_/11_`（3.11 环境）、`12_/13_`（按批次加载）。

**备用工具**：`label_tool.py`（零依赖，标准库 http.server + 浏览器 Canvas），
在补丁排查空档期用于不中断标注进度。注意两个工具都是整份重写 txt，**不要同时对同一批图片标注**。

### 4.3 标注方式（如实说明）

标注采用 **「模型预标注 + 人工复核」**，**并非全部逐框手工绘制**：

1. 先用 COCO 预训练模型 `yolo11n.pt` 生成草稿（`auto_label.py`，映射 bottle→cola、sports ball→football）；
2. 再用第一版模型做主动学习式二次预标注（`auto_annotate_rest.py`），
   并按类别阈值精修与跨类冲突消解（`refine_auto_labels.py`）：
   obstacle ≥ 0.45（弱类从严）、cola ≥ 0.40、football ≥ 0.30，不同类别 IoU > 0.5 时丢弃低置信度框；
3. 最后逐张人工复核：删除误框、补画漏框、补画 obstacle。

**标注规范**：框贴紧目标外轮廓（留白 < 目标尺寸 5%）；被遮挡面积 >70% 不标，30%~70% 只框可见部分；
最小边长 <1% 图宽不标；同一目标只画一个框；禁止整图框。

### 4.4 标注进度（诚实记录）

| 来源目录 | 图片数 | 含该类目标框 | 覆盖率 |
| --- | --- | --- | --- |
| football | 340 | 306 | 90.0% |
| cola | 293 | 226 | 77.1% |
| **obstacle** | 316 | 104 | **32.9%** |

**尚未完成的工作**（`audit_labels.py` 与 `check_labels.py` 的实际输出）：

- **96 张图片没有任何标签文件**（train 79 + val 17），不满足"每张参与训练的图片都应具有对应的标签文件"；
- **209 张图片待补 obstacle**（`label_plan.csv`，分 6 批：P1 152 / P2 47 / P3 10），目前完成 0 张；
- `check_labels.py` 当前返回 **FAIL**（就是上述无标签图片导致）。

这直接反映在模型指标上：obstacle 的 mAP50 仅 **0.647**（召回 0.565），明显低于 cola 0.962、football 0.866。
后续把 obstacle 补齐后重训，该项指标应明显提升。

### 4.5 标注质量自检

| 脚本 | 作用 |
| --- | --- |
| `check_labels.py` | 图片与标签配对、字段数、坐标范围 0~1、类别编号合法性、三类均有实例 |
| `audit_labels.py` | 基于来源目录先验的漏标排查、重复框、跨类冲突、超小框、整图框；生成待办清单 |
| `batch_progress.py` | 按批次统计标注进度 |
| `verify_structure.py` | 数据集目录结构校验 |

---

## 五、模型训练（Level 3）

### 5.1 数据集配置文件（Level 3 第 1 节）

`data.yaml` 内容：

```yaml
path: ./dataset          # 数据集根目录
train: images/train      # 训练集图片目录（相对 path）
val: images/val          # 验证集图片目录（相对 path）

nc: 3                    # 类别数量，与 names 条数一致
names:
  0: obstacle
  1: cola
  2: football
```

校验脚本 `verify_data_yaml.py` 的四项检查全部通过（退出码 0）：

```text
1) 解析 YAML            [PASS] 含字段 path / train / val / names；类别编号与名称正确
2) Ultralytics 解析路径  解析结果 train = ...\yolo_project\dataset\images\train
                        解析结果 val   = ...\yolo_project\dataset\images\val
                        [PASS] train / val 指向的目录存在
3) 数据规模（当前）      [train] 图片 755 张，标签 676 个，格式异常行 0
                               obstacle=94  cola=253  football=623
                        [val]   图片 188 张，标签 171 个，格式异常行 0
                               obstacle=27  cola=58  football=154
4) 与 classes.txt 一致性 [PASS] ['obstacle', 'cola', 'football']
结论: [PASS] data.yaml 配置正确，可以直接用于训练
```

> 说明：本次训练执行时数据集为 train 760 张 / val 189 张；之后为 Level 4 严格评测主动移出 6 张
> （5 张来自 train、1 张来自 val），因此当前统计为 755 / 188。`data.yaml` 只指向目录、不含张数，
> 移出图片不影响配置的正确性。

> **路径解析规则**（已对照 ultralytics 源码与实测确认）：`path` 写相对路径时，
> ① 先看**当前工作目录**下是否存在该相对路径，存在就直接用；
> ② 不存在才回退到它自己的 `datasets_dir` 下找。
> 因此运行 `train.py` / `predict.py` 时必须在项目目录下执行。

另外提供 `data_subset.yaml`（只取已标注图片的清单文件），用于冒烟测试跑通流程。

### 5.2 训练参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 模型 | `yolo11n.pt` | 官方预训练权重做迁移学习（小数据集 + 4 GB 显存） |
| epochs | 100 | 本次 100 轮跑满，未触发早停 |
| imgsz | 640 | 原图 640×480 |
| batch | 8 | 4 GB 显存安全值 |
| optimizer | auto | 配合 lr0=0.01 |
| cache | disk | RAM 缓存会在 Windows 上 pickle 失败，故用磁盘 |
| amp | False | 混合精度自检需联网下载权重，本机网络不通会卡住 |
| close_mosaic | 10 | 最后 10 轮关闭 mosaic 增强 |
| seed / deterministic | 42 / True | 保证结果可复现 |
| patience | 25 | 连续 25 轮无提升则早停 |

完整参数见 `runs/detect/train/args.yaml`（115 行）。训练命令：`python train.py`。

### 5.3 训练结果

- 训练轮数：**100 轮**（跑满）；总耗时 **0.428 小时（约 25.7 分钟）**
- 权重：`runs/detect/train/weights/best.pt`，5,493,457 字节（约 5.24 MB），参数量 **2,590,425**
- `best.pt` 对应第 **82** 轮（Ultralytics 按 fitness 选择；本版本 fitness 权重为 `[0,0,0,1]`，即等于 mAP50-95）

| 轮次 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| 第 1 轮 | 0.897 | 0.265 | 0.457 | 0.354 |
| 第 59 轮 | 0.876 | 0.756 | 0.832 | 0.709 |
| **第 82 轮（best.pt）** | **0.831** | **0.781** | **0.825** | **0.710** |
| 第 100 轮 | 0.837 | 0.740 | 0.805 | 0.700 |

训练后单独验证（`model.val()`）：

| 类别 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| obstacle | 0.841 | 0.565 | 0.647 | 0.541 |
| cola | 0.856 | 0.932 | 0.962 | 0.909 |
| football | 0.798 | 0.846 | 0.866 | 0.678 |
| **all** | **0.832** | **0.781** | **0.825** | **0.710** |

混淆矩阵（行 = 预测，列 = 真实；阈值 conf=0.25 / IoU=0.45，验证集 188 张 / 239 个实例）：

| pred ↓ / true → | obstacle | cola | football | background（误检） |
| --- | --- | --- | --- | --- |
| obstacle | 15 | 0 | 3 | 7 |
| cola | 0 | 57 | 0 | 13 |
| football | 3 | 0 | 139 | 40 |
| background（漏检） | 9 | 1 | 12 | 0 |

> 方向自检：各**列**求和恰等于该类真实实例数（27 / 58 / 154），与数据集统计一致。
> 说明：混淆矩阵用固定阈值 conf=0.25，而终端报告的 P / R 取 PR 曲线 F1 最大点，两者工作点不同，不宜直接互比。

### 5.4 训练产物

`runs/detect/train/` 下：`weights/best.pt`、`weights/last.pt`、`results.png`、`results.csv`（101 行）、
`confusion_matrix.png`、`confusion_matrix_normalized.png`、`BoxPR_curve.png`、`BoxP/R/F1_curve.png`、
`labels.jpg`、`train_batch*.jpg`、`val_batch*_labels.jpg`、`val_batch*_pred.jpg`、`args.yaml`。

同目录下另有：`train_holdout`（Level 4 防泄漏重训）、`bootstrap`（预标注自举模型）、
`val` / `val-2` / `val_final`（单独验证记录）。

---

## 六、新图片目标检测（Level 4，及格线）

### 6.1 新图片的准备（保证"训练集之外"）

原始 949 张图片**已全部进入数据集，没有富余图片**。因此从数据集中主动移出 **6 张**
（5 张来自 train、1 张来自 val）作为新图片：

- 移出记录：`test_images/_manifest.txt`
- 移出前备份：`dataset/_holdout_backup_20260924_151813/`
- 移出脚本：`make_holdout_set.py`

为保证这 6 张**从未参与训练与验证**，重新训练了排除它们的模型
（`runs/detect/train_holdout/weights/best.pt`，数据集 755/188）。本节全部推理与评测都使用该权重。

### 6.2 第一步：加载训练模型

`python load_model.py` → `results/evidence_level4/01_加载模型.txt`

脚本从权重文件**内部**读出训练履历并逐条判定，**8 项检查全部 PASS**：

| 检查项 | 结果 |
| --- | --- |
| 权重文件指纹 | 5,493,457 字节 / SHA256 `d74282b0…98d0a` |
| 任务 / 类别数 | detect / 3 |
| 类别名称 | obstacle、cola、football |
| checkpoint 记录的 `data` | `D:\研一\git 练习\yolo_project\data.yaml` |
| checkpoint 记录的 `model` / epochs / imgsz / batch | yolo11n.pt / 100 / 640 / 8 |
| 与 `results.csv` 交叉验证 | 完全对应第 82 轮（最大误差 0） |
| 与官方 COCO 权重对比 | 官方 80 类 vs 本模型 3 类 |

### 6.3 第二步：对新图片推理

`python infer_and_check.py` → `results/evidence_level4/02_对新图片推理.txt`

对 6 张新图片推理（`imgsz=640`、`conf=0.25`），与随图带出的真值逐目标比对
（IoU≥0.5 且类别相同记为命中）：

| 类别 | TP | FP | FN | 精确率 | 召回率 |
| --- | --- | --- | --- | --- | --- |
| obstacle | 4 | 0 | 0 | 1.000 | 1.000 |
| cola | 3 | 0 | 0 | 1.000 | 1.000 |
| football | 10 | 2 | 4 | 0.833 | 0.714 |
| **合计** | **17** | **2** | **4** | **0.895** | **0.810** |

**能力边界分析**：4 个漏检的足球真值框面积均在 0.0003~0.0009（小于整图千分之一），
2 个误报的置信度仅 0.277 / 0.298；而所有命中目标的框面积都 ≥0.0025。
说明问题集中在**极小目标**，而非类别理解错误。

### 6.4 第三步：保存检测结果

`python save_detection_results.py` → `results/evidence_level4/04_保存检测结果.txt`

检测结果保存为**四种互相印证**的形式：

| 形式 | 路径 | 作用 |
| --- | --- | --- |
| ① 画框结果图 | `results/saved_results/new_images/*.jpg` | 显示目标框 + 类别名称 + 置信度 |
| ② YOLO 格式标签 | `.../new_images/labels/*.txt` | 每行 6 列：类别编号 cx cy w h 置信度 |
| ③ 目标裁剪图 | `.../new_images/crops/<类别名>/*.jpg` | 按类别名分目录，每个目标一张 |
| ④ 结构化汇总表 | `results/evidence_level4/05_检测结果汇总表.csv` | 22 行目标记录，Excel 可打开 |

一致性校验 4 项全部 PASS：标签行数 = 检出目标数（19）；标签内容与预测逐项一致（容差 1e-5）；
裁剪图数量 = 检出目标数；画框结果图与原图逐像素比对确实存在差异。

### 6.5 三种展示情况

| 情况 | 效果图 |
| --- | --- |
| ① 单个目标检测 | `results/evidence_level4/06_情况①单目标检测.jpg`（obstacle / cola / football 各一张） |
| ② 多个目标同时检测 | `results/evidence_level4/07_情况②多目标检测.jpg`（6 张新图片） |
| ③ 不同场景与拍摄角度 | `results/evidence_level4/08_情况③不同场景与角度.jpg`（三个不同拍摄时段） |
| 整体效果图 | `results/evidence_level4/03_检测效果图_拼接.jpg`（1338×1188） |

### 6.6 验证标准逐条对照（5 / 5 通过）

`python check_level4.py` → `results/evidence_level4/09_Level4验收对照表.txt`（退出码 0）

| # | 考核验证标准 | 结论 |
| --- | --- | --- |
| 1 | 正确加载自己训练得到的模型 | ✅ 通过（8 项检查 PASS） |
| 2 | 对若干张训练集之外的新图片完成推理 | ✅ 通过（6 张，确认不在 dataset 内） |
| 3 | 保存目标检测结果 | ✅ 通过（四种形式，数量互相一致） |
| 4 | 提交至少一组清晰的最终检测效果图 | ✅ 通过（4 张效果图） |
| 5 | 结果中能显示目标框、类别名称和置信度 | ✅ 通过（6 列标签 + 按类别名分目录的裁剪图 + 画框验证） |

---

## 七、遇到的问题与解决（摘要）

| # | 问题 | 解决 |
| --- | --- | --- |
| 1 | PATH 里的 `python` 是应用商店别名，报"系统找不到指定的路径" | 改用虚拟环境解释器全路径 |
| 2 | LabelImg 与新版 PyQt5 不兼容，按 `W` 就闪退（`0xC0000409` 无报错） | 写 6 个幂等补丁脚本（float→int、assert 改优雅处理）；另建 Python 3.11 备用环境 |
| 3 | 项目路径含中文与空格导致部分工具异常 | 虚拟环境装到 ASCII 路径；脚本统一用 `Path(__file__).resolve().parent` 定位 |
| 4 | 4 GB 显存上限，预标注时 CUDA OOM | 预标注改 CPU 推理；训练 batch=8 |
| 5 | 无标签图片会被当作背景图参与训练，导致缺失类别学不到 | `train.py` 加入训练前预检，缺类别直接中止并提示 |
| 6 | 两次划分脚本结果不一致 | 统一以 `split_dataset.py`（seed=42）为唯一入口，已标标签按实际 split 重新归位 |
| 7 | `cache=True`（RAM）在 Windows 上 pickle 失败 | 改 `cache='disk'` |
| 8 | AMP 自检联网下载权重卡住训练启动 | 设 `amp=False` |
| 9 | 受限终端下 DataLoader 报 `PermissionError [WinError 5]` | 用管理员终端运行训练 |
| 10 | 混淆矩阵方向易读错 | 核对 ultralytics 源码确认 `matrix[预测, 真实]`，并用列求和自检 |
| 11 | Ultralytics 每次校验都要联网下载字体导致检查失败 | 把系统字体放入其配置目录，实现完全离线校验 |

---

## 八、目录与文件说明

```text
123456789/
├── README.md                 本文件（提交说明）
├── report.md                 项目报告（9 个章节，含各 Level 验证材料）
├── classes.txt               类别定义：obstacle / cola / football
├── data.yaml                 数据集配置
├── data_subset.yaml          子集配置（冒烟测试用）
├── requirements.txt          依赖版本
├── train.py                  训练脚本（含训练前类别预检）
├── predict.py                基础推理脚本
├── split_dataset.py          数据集划分（分层 8:2，seed=42）
├── verify_data_yaml.py       数据集配置校验
├── check_labels.py           标注自检
├── audit_labels.py           标注质量审计
├── auto_label.py             COCO 预标注
├── auto_annotate_rest.py     自举模型二次预标注
├── refine_auto_labels.py     按类别阈值精修预标注
├── make_label_plan.py        生成分批标注计划
├── batch_progress.py         标注进度统计
├── label_tool.py             零依赖备用标注工具
├── load_model.py             Level 4-1 加载模型（8 项检查）
├── infer_and_check.py        Level 4-2 新图片推理与评测
├── save_detection_results.py Level 4-3 保存检测结果（四种形式）
├── check_level4.py           Level 4 验收标准自动核对
├── make_holdout_set.py       移出 Level 4 测试图片
├── make_level3_evidence.py   Level 3 证据生成
├── _read_metrics.py          读取混淆矩阵与分类别指标
├── _patch_labelimg.py 等      LabelImg 兼容性补丁（6 个）
├── *.bat                     一键启动脚本（1~14 号）
├── dataset/                  数据集（943 张图 + 847 个标签）
├── test_images/              6 张新图片 + 真值 + _manifest.txt
├── runs/detect/              训练输出（train / train_holdout / bootstrap / val*）
├── results/                  全部验收证据、效果图、CSV 汇总表
├── label_batches/            分批标注清单
└── labels_backup/            标签备份
```

### 验收证据索引

| 证据文件 | 对应内容 |
| --- | --- |
| `results/evidence_level3/01~05*.txt` | Level 3：训练启动 / 结果目录 / 指标汇总 / 权重信息 / 验证与测试 |
| `results/evidence_level4/01_加载模型.txt` | Level 4-1 |
| `results/evidence_level4/02_对新图片推理.txt` | Level 4-2 |
| `results/evidence_level4/03~08*.jpg` | 检测效果图（整体 + 三种展示情况） |
| `results/evidence_level4/04_保存检测结果.txt` | Level 4-3 与一致性校验 |
| `results/evidence_level4/05_检测结果汇总表.csv` | 逐目标结构化结果 |
| `results/evidence_level4/09_Level4验收对照表.txt` | Level 4 五条验证标准逐条对照 |

---

## 九、一键复现

```powershell
# 0. 进入项目目录（data.yaml 用相对路径，必须在项目目录下运行）
cd 123456789

# 1. 环境（已装好可跳过）
.\setup_env.ps1

# 2. 校验数据集配置（应输出 PASS）
C:\Users\dell\yolo-env\Scripts\python.exe verify_data_yaml.py

# 3. 训练（约 25 分钟；先冒烟测试更快：加 --epochs 2 --subset）
C:\Users\dell\yolo-env\Scripts\python.exe train.py

# 4. Level 4 四个环节（依次执行，全部退出码 0）
C:\Users\dell\yolo-env\Scripts\python.exe load_model.py
C:\Users\dell\yolo-env\Scripts\python.exe infer_and_check.py
C:\Users\dell\yolo-env\Scripts\python.exe save_detection_results.py
C:\Users\dell\yolo-env\Scripts\python.exe check_level4.py

# 5. 标注质量自检
C:\Users\dell\yolo-env\Scripts\python.exe check_labels.py
C:\Users\dell\yolo-env\Scripts\python.exe audit_labels.py
```

---

## 十、已知不足与后续计划

**不足**：

1. **`obstacle` 类标注不足**：覆盖率仅 32.9%（209 张待补），且 96 张图片没有标签文件，
   `check_labels.py` 当前为 FAIL；这直接导致 obstacle 的 mAP50（0.647）远低于 cola（0.962）与 football（0.866）；
2. **标注边界问题未统一**：部分"橙色球 / 橙色方块"究竟算 `obstacle` 还是 `football` 仍需全局统一后重训；
3. **极小目标召回不足**：Level 4 的 4 个漏检、2 个误报全部集中在面积 <0.001 的目标上；
4. **标注来源不完全统一**：部分标签由模型预标注生成，与纯人工标注在框的松紧度上存在细微差异；
5. **验证集偏小**：188 张 / 239 个实例，单轮指标抖动较大，指标只宜作趋势参考；
6. **Level 5 未完成**：本地图片识别页面尚未开始。

**后续计划**：

1. 补齐 209 张 obstacle 标注并处理 96 张无标签图片，重跑 `check_labels.py` 至 PASS；
2. 统一 obstacle 标注边界，重新训练模型并刷新 Level 3 / Level 4 全部证据；
3. 尝试更高训练分辨率或切片推理（SAHI 思路）改善极小目标；
4. 用 Streamlit / Gradio 完成 Level 5 本地识别页面。
