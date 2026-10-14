# 基于 YOLO 的机器人场景目标检测 · 项目报告

- 姓名：C
- 学号：123456789
- 专业班级：待填
- 项目：2026 工程实践能力考核 · 项目一（YOLO）
- 目标等级：Level 4（及格线）

---

## 一、环境配置（对应 Level 1）

| 组件 | 版本 / 说明 |
| --- | --- |
| 操作系统 | Windows |
| 显卡 | NVIDIA GeForce RTX 3050 Laptop GPU（4GB 显存），驱动 610.88 |
| Python | 3.14.7（py 启动器管理） |
| 虚拟环境 | `C:\Users\dell\yolo-env`（ASCII 路径，避免中文/空格路径问题） |
| PyTorch | 2.14.0+cu126 |
| TorchVision | 0.29.0+cu126 |
| Ultralytics | 8.4.155 |
| OpenCV | 5.0.0 |

安装与验证命令：

```powershell
py -3.14 -m venv C:\Users\dell\yolo-env
C:\Users\dell\yolo-env\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
C:\Users\dell\yolo-env\Scripts\python.exe -m pip install ultralytics
C:\Users\dell\yolo-env\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

验证输出（截图 1：环境安装与 CUDA 可用性）：

```text
2.14.0+cu126 True NVIDIA GeForce RTX 3050 Laptop GPU
ultralytics 8.4.155
```

> 截图 1：终端中 Python / PyTorch / Ultralytics 安装成功与 `torch.cuda.is_available() = True`

---

## 二、原始数据检查（对应 Level 1）

| 数据集 | 图片数量 | 格式 | 分辨率 |
| --- | --- | --- | --- |
| obstacle | 316 | jpg | 640×480（部分 640×400） |
| cola | 293 | jpg | 640×480 |
| football | 340 | jpg | 640×480 |
| 合计 | 949 | — | — |

检查结论：所有图片均可正常读取，无损坏文件。

> 截图 2：原始数据统计（三类图片数量、格式与分辨率）

---

## 三、数据集划分（对应 Level 1）

本项目将实验室提供的 949 张原始图片按 **类别分层随机划分** 组织为 YOLO 数据集：

- 划分比例：训练集 : 验证集 = 8 : 2
- 随机种子：42（固定种子，保证划分结果可复现）
- 划分脚本：`split_dataset.py` —— 对 obstacle / cola / football 三个类别分别独立打乱后按比例切分，
  避免某一类别在验证集中缺失
- 划分结果：

| 类别 | 原始张数 | 训练集 | 验证集 |
| --- | --- | --- | --- |
| obstacle | 316 | 253 | 63 |
| cola | 293 | 235 | 58 |
| football | 340 | 272 | 68 |
| **合计** | **949** | **760** | **189** |

- 目录对应关系：`dataset/images/<split>` 与 `dataset/labels/<split>` 中的文件一一对应，
  同一张图片的标签与图片主文件名相同（如 `image_rgb_20260727_143017.jpg` ↔
  `image_rgb_20260727_143017.txt`）
- 标签格式：YOLO 格式，每行 5 个值 = 类别编号 + 归一化中心点 x/y + 归一化宽/高

### 3.1 项目目录结构（`show_structure.py` 实际输出）

```text
yolo_project/
|-- dataset/                        [1168 files]
|   |-- images/                      [949 files]
|   |   |-- train/                   [760 files]
|   |   +-- val/                     [189 files]
|   +-- labels/                      [219 files]
|       |-- train/                   [178 files]
|       +-- val/                     [ 41 files]
|-- runs/                            [训练输出目录]
|-- results/                         [推理结果目录]
|-- test_images/                     [Level 4 新图片]
|-- data.yaml                        [数据集配置]
|-- classes.txt                      [标注类别顺序]
|-- train.py                         [训练脚本]
|-- predict.py                       [推理脚本]
|-- split_dataset.py                 [数据划分]
|-- label_tool.py                    [零依赖标注工具]
|-- auto_label.py                    [预标注辅助]
|-- check_labels.py / audit_labels.py [标注质量自检]
|-- batch_progress.py                [标注进度跟踪]
|-- requirements.txt                 [依赖清单]
|-- README.md / report.md            [说明与报告]
```

> 截图 3：`python show_structure.py` 的终端输出（即上面的目录结构，含各目录文件数）
> 截图 4：`python check_labels.py` 的输出（train 760 / val 189 与三类实例统计，佐证划分与标签对应关系）

### 3.2 划分脚本的确定（过程记录）

Level 1 第 3 步曾先后用过两个划分脚本，划分结果（成员名单）不同，最终统一以
项目内的 `split_dataset.py` 为准：

| 脚本 | 随机方式 | cola 验证集 | train / val 合计 |
| --- | --- | --- | --- |
| `split_dataset.ps1`（早期试写） | .NET `System.Random` + 手写 Fisher-Yates | 59 | 759 / 190 |
| `split_dataset.py`（**最终采用**） | `random.seed(42)` + `random.shuffle` | 58 | **760 / 189** |

两者都是“每类独立打乱后取前 20% 作验证集”，差异来自 `int(293*0.2)` 的截断（58）与
`[int](293*0.2)` 的进位（59），以及两种随机数发生器生成的下标序列不同。

最终用 `split_dataset.py` 重新生成数据集（先清空 `dataset/`，再重新划分），并做了两件事：

1. 旧划分下已有的标签文件按“图片现在所在的 split”重新归位，共 219 个标签无丢失、无孤儿
   （其中 83 个标签因图片换到了另一个 split 而随之移动）；
2. 重新执行 `verify_structure.py`，结果 **PASS**（结构、data.yaml 字段与类别名、图片/标签配对、多余标签 = 0）。

> 说明：`split_dataset.ps1` 属于早期尝试，重复运行会再次改变划分结果，以 `split_dataset.py` 为唯一入口。

#### 3.2.1 各类别当前标注覆盖情况（重新划分后实测）

| split | 类别 | 图片数 | 已标注图片 | 待标注 |
| --- | --- | --- | --- | --- |
| train | obstacle | 253 | 40 | 213 |
| train | cola | 235 | 105 | 130 |
| train | football | 272 | 33 | 239 |
| val | obstacle | 63 | 8 | 55 |
| val | cola | 58 | 25 | 33 |
| val | football | 68 | 8 | 60 |
| **合计** | — | **949** | **219** | **730** |

---

## 四、目标标注（对应 Level 2）

### 4.1 类别定义

| 编号 | 类别 | 含义 | 判定规则 |
| --- | --- | --- | --- |
| 0 | obstacle | 障碍物 | 机器人行进路径上有碰撞风险、且不属于 cola / football 的实体物体；排除阴影、反光、贴纸、纯背景区域 |
| 1 | cola | 可乐瓶 | 可乐瓶本身；该区域不再标 obstacle |
| 2 | football | 足球 | 足球本身；该区域不再标 obstacle |

标注规范：框贴紧目标外轮廓（留白 < 目标尺寸 5%）；被遮挡面积 >70% 不标，30%~70% 只框可见部分；
最小边长 <1% 图宽不标；同一目标只画一个框；禁止整图框。

### 4.2 标注工具

本项目使用两种工具完成标注：

**1. LabelImg 1.8.6（主要工具，考核推荐）**

- **两套可切换的安装环境**（标注环境与训练环境分开，一个环境出问题时可以换另一个继续标）：

  | 环境 | Python | labelImg | PyQt5 | 对应启动脚本 |
  | --- | --- | --- | --- | --- |
  | `C:\Users\dell\yolo-env` | 3.14.7 | 1.8.6 | 5.15.11（Qt 5.15.2 / sip 12.19.0） | `7_`、`8_` |
  | `C:\Users\dell\label-env` | 3.11.9 | 1.8.6 | 5.15.9（sip 12.13.0） | `10_`、`11_`、`12_`、`13_` |

  安装命令均为 `pip install labelImg==1.8.6`，会自动带上 PyQt5 与 `lxml 6.1.3`。
  过程说明：最初判断“Python 3.14 装不上 LabelImg”（担心 PyQt5 无对应轮子），因此先装了
  Python 3.11.9（`C:\Users\dell\py311`）并建立专用标注环境 `C:\Users\dell\label-env`；
  后实测发现 PyQt5 提供 `cp38-abi3` 轮子，在 3.14 上同样装得上、图形界面也能开，
  于是训练环境 `yolo-env` 里也装了一份。两套环境都保留，均可用于标注。

- **必须打补丁才能稳定使用**：labelImg 1.8.6 是 2021 年的版本，与新版 PyQt5 不兼容，
  会出现“按 `W` 画框即闪退”（退出码 `0xC0000409`，且屏幕上没有任何报错）。
  共 6 个幂等补丁脚本（可重复运行）：

  | 补丁脚本 | 修复内容 |
  | --- | --- |
  | `_patch_labelimg.py` | `QScrollBar / QSpinBox.setValue()` 收到 float → 显式转 `int()` |
  | `_fix_labelimg_painter.py` | `QPainter.drawRect / drawLine / drawText` 收到 float → 显式转 `int()`（按 `W` 拖动闪退的真正原因） |
  | `_fix_labelimg_w_key.py` | `W` 被两个动作重复绑定 → 释放快捷键冲突 |
  | `_fix_labelimg_asserts.py` | `create_shape / paint_canvas / save_file_as` 中的 assert → 改为优雅处理 |
  | `_fix_libs_asserts.py` | `canvas.py` / `shape.py` 中会把进程直接搞死的 assert → 改为优雅处理 |
  | `_fix_classes_txt.py` | 把 `classes.txt` 复制进标签目录（YoloReader 依赖），并让自检脚本跳过它 |

  补充实测：PyQt5 5.15.6 / 5.15.9 + sip 12.11~12.13 **同样拒收 float**，5.15.2 / 5.15.4 无对应轮子，
  因此“免补丁即可用”的版本组合不存在，补丁是必需步骤。

- **类别文件配置**：`classes.txt` 必须放两处 ——

  1. 项目根 `classes.txt`：作为启动的第二个参数传入，决定界面可选的类别；
  2. `dataset/labels/train/classes.txt` 与 `dataset/labels/val/classes.txt`：labelImg 的 YoloReader
     读取已有标签时会去**标签所在目录**找它（`libs/yolo_io.py`），缺失会在启动时抛 `FileNotFoundError`。

  内容固定三行：`obstacle` / `cola` / `football`，顺序即类别编号 0 / 1 / 2。

- **启动命令（三参数写法）**：第三个参数 `save_dir` 让标注结果直接写进 `dataset/labels/<split>/`，
  无需事后搬运（若省略，txt 会落在图片目录，需用 `fix_label_placement.py` 归位）。

  ```powershell
  # 标注训练集
  C:\Users\dell\yolo-env\Scripts\labelImg.exe "D:\研一\git 练习\yolo_project\dataset\images\train" "D:\研一\git 练习\yolo_project\classes.txt" "D:\研一\git 练习\yolo_project\dataset\labels\train"
  # 标注验证集
  C:\Users\dell\yolo-env\Scripts\labelImg.exe "D:\研一\git 练习\yolo_project\dataset\images\val" "D:\研一\git 练习\yolo_project\classes.txt" "D:\研一\git 练习\yolo_project\dataset\labels\val"
  ```

  已封装成双击即可运行的启动脚本（用 `%~dp0` 自动定位项目目录，`chcp 65001` 处理中文路径）：

  | 启动脚本 | 使用环境 | 标注范围 |
  | --- | --- | --- |
  | `7_labelImg_train.bat` / `8_labelImg_val.bat` | yolo-env（Py3.14） | train / val 全部图片 |
  | `10_labelImg311_train.bat` / `11_labelImg311_val.bat` | label-env（Py3.11） | train / val 全部图片 |
  | `12_labelImg_b01_train.bat` / `13_labelImg_b01_val.bat` | label-env（Py3.11） | 仅 batch_01 子集（图片先复制到 `label_batches/_stage/`，避免在 900 多张图里滚动查找） |

- 首次使用需要设置：左侧工具栏把保存格式切到 **YOLO**（否则会存成 PascalVOC 的 xml）；
  类别取自 `classes.txt`，顺序为 0 obstacle / 1 cola / 2 football；
  常用快捷键：`W` 画框、`Ctrl+S` 保存、`A`/`D` 上一张/下一张、`Ctrl+D` 复制上一个框。

**2. label_tool.py（自研零依赖辅助工具）**

- 仅使用 Python 标准库 `http.server` + 浏览器 Canvas，标签直接写入 `dataset/labels/<split>/`，
  自带批次清单与进度显示，适合批量补框与查漏。

> 注意事项：两个工具保存时都是整份重写 txt，**不要同时对同一批图片标注**，切换工具前先保存并关闭另一个；
> 若使用 LabelImg 时未指定第三个参数，txt 会落在图片目录，需要执行归位命令搬到 `dataset/labels/<split>/`。

### 4.3 预标注辅助与人工复核

为提升效率，先用 COCO 预训练模型 `yolo11n.pt` 做了一遍预标注（`auto_label.py`，
bottle→cola、sports ball→football），再由人工逐张复核、删除误框、补画 obstacle。
预标注结果与人工复核量如实记录如下：

| 数据集 | 预标注生成标签 | 实例数（cola / football） | 仍需人工处理 |
| --- | --- | --- | --- |
| train（760 张） | 178 | 85 / 106 | 582 |
| val（189 张） | 41 | 20 / 22 | 148 |

按图片来源目录统计的复核优先级见 `label_worklist.csv`；obstacle 类别 COCO 中没有对应类，全部由人工标注。

> 截图 5：标注工具界面 / 已完成标注的图片
> 截图 6：某个 YOLO 标签文件内容（形如 `2 0.662941 0.580592 0.148862 0.183283`）

### 4.4 标注质量自检

- `check_labels.py`：图片与标签配对、字段数、坐标范围、类别编号合法性
- `audit_labels.py`：基于来源目录先验的漏标排查、重复框、跨类冲突、超小框、整图框
- `batch_progress.py`：按批次统计标注进度

> 截图 7：自检脚本输出（class 0/1/2 实例数 与 PASS 结论）

---

## 五、模型训练（对应 Level 3）

### 5.0 数据集配置文件 data.yaml（Level 3 第 1 节）

`data.yaml` 用于告诉 YOLO “图片在哪、分为哪几类”。本项目配置如下：

```yaml
path: ./dataset          # 数据集根目录（相对路径的解析规则见下方备注）
train: images/train      # 训练集图片目录（相对 path）
val: images/val          # 验证集图片目录（相对 path）

nc: 3                    # 类别数量
names:
  0: obstacle
  1: cola
  2: football
```

校验脚本 `verify_data_yaml.py` 的实际输出（确认路径能被 Ultralytics 正确解析）：

```text
1) 解析 YAML
  [PASS] 含字段 path / train / val / names
  [PASS] 类别编号与名称正确（0 obstacle / 1 cola / 2 football）

2) Ultralytics 实际解析出的路径
  解析结果 train = D:\研一\git 练习\yolo_project\dataset\images\train
  解析结果 val   = D:\研一\git 练习\yolo_project\dataset\images\val
  解析结果 nc    = 3
  [PASS] train / val 指向的目录存在

3) 数据规模与类别统计（当前数据集）
  [train] 图片 755 张，标签 676 个，格式异常行 0
         obstacle=94  cola=253  football=623
  [val] 图片 188 张，标签 171 个，格式异常行 0
         obstacle=27  cola=58  football=154

4) 与 classes.txt 一致性
  [PASS] classes.txt = ['obstacle', 'cola', 'football']

结论：[PASS] data.yaml 配置正确，可以直接用于训练
```

> 备注（路径解析规则，已对着 ultralytics 源码与实测确认）：
> `data.yaml` 里 `path` 写成相对路径时，Ultralytics 的解析顺序是
> ① 先看**当前工作目录**下是否存在该相对路径（`./dataset`），存在就直接用；
> ② 不存在才回退到它自己的 `datasets_dir`（本机为 `D:\研一\git 练习\datasets`）下找。
> 因此本项目必须在 `yolo_project` 目录下运行 `train.py` / `predict.py` 等脚本；
> 换到其他目录运行会报 `dataset not found`（报错里会指向 `datasets\dataset\images\val`，属回退路径）。
> 这也是新手最常踩的 "dataset not found" 根源。
>
> 截图 7.1：`verify_data_yaml.py` 的终端输出（data.yaml 配置正确的证据）

### 5.1 训练参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 模型 | yolo11n.pt | 预训练权重（小数据集 + 4GB 显存） |
| epochs | 100 | patience=25 早停（本次 100 轮跑满，未触发早停） |
| imgsz | 640 | 原图 640×480 |
| batch | 8 | 4GB 显存安全值（不足时降为 4） |
| optimizer | auto | 小数据集自动选择，配合 lr0=0.01 |
| cache | disk | 磁盘缓存。RAM 缓存会把图像数组带进 DataLoader 子进程，Windows 上会 pickle 失败，故改用磁盘 |
| workers | 4 | 数据加载子进程数 |
| amp | False | 关闭混合精度自检（自检会尝试联网下载 yolo26n.pt，本机网络不可达会卡在启动阶段） |
| close_mosaic | 10 | 最后 10 轮关闭 mosaic 增强，使模型在真实图像分布上收敛 |
| seed | 42 | 保证可复现（deterministic=True） |

训练命令：`python train.py`（详细参数见 `runs/detect/train/args.yaml`，共 115 行）

训练产出（`runs/detect/train/`）：`weights/best.pt`、`weights/last.pt`、`results.png`、
`results.csv`、`confusion_matrix.png`、`confusion_matrix_normalized.png`、
`BoxPR_curve.png`、`BoxP_curve.png`、`BoxR_curve.png`、`BoxF1_curve.png`、
`val_batch*_pred.jpg`、`val_batch*_labels.jpg`、`args.yaml`

`best.pt` 对应第 **82** 轮（按 fitness = mAP50-95 选中）：P 0.831 / R 0.781 / mAP50 0.825 / mAP50-95 0.710。
末轮（第 100 轮）为 P 0.837 / R 0.740 / mAP50 0.805 / mAP50-95 0.700。
训练总耗时 0.428 小时（约 25.7 分钟），权重 5,493,457 字节（约 5.24 MB），参数量 2,590,425。

> 截图 8：训练启动终端输出
> 截图 9：训练结果目录
> 截图 10：results.png（Loss / Precision / Recall / mAP 曲线）
> 截图 11：best.pt 权重文件信息

### 5.2 指标解读

| 指标 | 含义 | 判断标准 |
| --- | --- | --- |
| box_loss / cls_loss / dfl_loss | 定位 / 分类 / 边框回归误差 | 前 10 轮快速下降后趋于平稳 |
| Precision | 报出的框中正确比例 | 0.7 以上较好 |
| Recall | 真实目标召回比例 | 0.6 以上较好 |
| mAP50 | IoU=0.5 平均精度 | >0.5 可用，>0.7 较好 |
| mAP50-95 | 更严格综合指标 | 约为 mAP50 的 55%~75% |
| 分类别 mAP | 每类精度分解 | 三个类别均不为 0 才算三类齐全 |

**实测补充**（由 `_read_metrics.py` 从 `best.pt` 直接读取；验证集 188 张图、239 个实例、17 张背景图）：

混淆矩阵（**行 = 预测 Predicted，列 = 真实 True**；最后一列 = 真实背景即误检，最后一行 = 预测为背景即漏检；阈值 conf=0.25 / IoU=0.45）：

| pred ↓ / true → | obstacle | cola | football | background（误检） |
| --- | --- | --- | --- | --- |
| obstacle | **15** | 0 | 3 | 7 |
| cola | 0 | **57** | 0 | 13 |
| football | 3 | 0 | **139** | 40 |
| background（漏检） | 9 | 1 | 12 | 0 |

方向自检：各**列**求和恰好等于该类真实实例数（obstacle 27、cola 58、football 154），
与数据集统计一致，可确认读法正确。由此得到：obstacle 漏检 9 / 误检 7；
cola 漏检 1 / 误检 13；football 漏检 12 / 误检 40。

其他观察：

- **mAP50-95 ÷ mAP50 = 0.703 ÷ 0.819 = 85.8%**，高于本表“55%~75%”的常见区间，
  说明边框定位精度较好，不只是“大致框住”。
- 混淆矩阵使用固定阈值 conf=0.25，而终端报告的 P / R 取 PR 曲线上 **F1 最大点**，
  两者工作点不同，数值不宜直接互相比较。
- 推理速度 3.4 ms/张（GPU），满足机器人实时检测需求。

---

## 六、新图片目标检测（对应 Level 4）

**本项目及格线要求：至少完成 Level 4 —— 使用自己训练得到的模型，对训练集之外的新图片完成目标检测。**

### 6.1 新图片的准备（保证“训练集之外”）

原始数据共 949 张（obstacle 316 / cola 293 / football 340），**已全部进入数据集，没有富余图片**。
因此从数据集中主动移出 6 张作为新图片（5 张来自 train、1 张来自 val）：

- 移出记录：`test_images/_manifest.txt`（逐张记录原属划分与类别）
- 移出前备份：`dataset/_holdout_backup_20260924_151813/`（可回滚）
- 移出脚本：`make_holdout_set.py`

为保证这 6 张图**从未参与训练与验证**，我重新训练了一个排除它们的模型
（`runs/detect/train_holdout/weights/best.pt`，数据集 755/188）。
本节全部推理与评测都使用该权重，而不是 `runs/detect/train/` 的权重
（那 6 张图仍在后者的训练集中，用它评测等于拿训练题当考试题）。

### 6.2 第 1 步：加载训练模型

执行 `python load_model.py`，输出 `results/evidence_level4/01_加载模型.txt`。

脚本不只看类别数，而是从权重文件**内部**读出训练履历并逐条判定，8 项检查全部 PASS：

| 检查项 | 结果 |
| --- | --- |
| 权重文件指纹 | 5,493,457 字节 / SHA256 `d74282b0…98d0a` |
| 任务 / 类别数 | detect / 3 |
| 类别名称 | obstacle、cola、football |
| checkpoint 记录的 `data` | `D:\研一\git 练习\yolo_project\data.yaml` |
| checkpoint 记录的 `model` | yolo11n.pt（迁移学习起点） |
| checkpoint 记录的 epochs/imgsz/batch | 100 / 640 / 8 |
| 与 `results.csv` 交叉验证 | 完全对应第 **82** 轮（最大误差 0） |
| 与官方 COCO 权重对比 | 官方 80 类 vs 本模型 3 类 |

关键结论：权重文件自身记录了“用哪个 `data.yaml`、训了多少轮、哪一轮最好”，
因此“这是我自己训练的模型”是可核验的事实，而不是陈述。

> 说明：Ultralytics 用 fitness 选择 `best.pt`，本版本 fitness 权重为 `[0,0,0,1]`，
> 即 fitness 完全等于 mAP50-95 —— 所以 `best.pt` 是 **mAP50-95 最高的第 82 轮**，
> 而不等于“mAP50 最高的第 59 轮”。

### 6.3 第 2 步：对新图片推理

执行 `python infer_and_check.py`，输出 `results/evidence_level4/02_对新图片推理.txt`。

对 6 张新图片推理（`imgsz=640`、`conf=0.25`），并与随图带出的真值标签逐目标比对
（预测框与真值框 IoU≥0.5 且类别相同记为命中）：

| 类别 | TP | FP | FN | 精确率 | 召回率 |
| --- | --- | --- | --- | --- | --- |
| obstacle | 4 | 0 | 0 | 1.000 | 1.000 |
| cola | 3 | 0 | 0 | 1.000 | 1.000 |
| football | 10 | 2 | 4 | 0.833 | 0.714 |
| **合计** | **17** | **2** | **4** | **0.895** | **0.810** |

**能力边界分析（数据得出）**：4 个漏检的足球真值框面积均在 0.0003~0.0009（小于整图的千分之一），
2 个误报的置信度仅 0.277 / 0.298；而所有命中目标的框面积都 ≥0.0025。
说明模型的问题集中在**极小目标**，而不是类别理解错误——obstacle 与 cola 的精确率、召回率均为 1.000。

### 6.4 第 3 步：保存检测结果

执行 `python save_detection_results.py`，输出 `results/evidence_level4/04_保存检测结果.txt`。

检测结果保存为**四种互相印证**的形式：

| 形式 | 路径 | 作用 |
| --- | --- | --- |
| ① 画框结果图 | `results/saved_results/new_images/*.jpg` | 显示目标框 + 类别名称 + 置信度（人眼看） |
| ② YOLO 格式标签 | `.../new_images/labels/*.txt` | 每行 6 列：类别编号 cx cy w h 置信度（机器可读） |
| ③ 目标裁剪图 | `.../new_images/crops/<类别名>/*.jpg` | 按类别名分目录，每个目标一张 |
| ④ 结构化汇总表 | `results/evidence_level4/05_检测结果汇总表.csv` | 22 行目标记录，Excel 可直接打开 |

一致性校验（4 项全部 PASS）：

- 标签文件行数 = 检出目标数（6 张图共 19 个目标）
- 标签文件内容（类别编号 + 置信度）与预测结果逐项一致（容差 1e-5）
- 裁剪图数量 = 检出目标数（19 张），按类别分布：football 12、obstacle 4、cola 3
- 画框结果图确实绘制了标注（与原图逐像素比较，6/6 张存在差异）

### 6.5 三种展示情况（Level 4 第 3 步建议）

| 情况 | 效果图 | 内容 |
| --- | --- | --- |
| ① 单个目标检测 | `results/evidence_level4/06_情况①单目标检测.jpg` | obstacle（conf 0.862）、cola（0.937）、football（0.900）各一张 |
| ② 多个目标同时检测 | `results/evidence_level4/07_情况②多目标检测.jpg` | 6 张新图片，每格标注检出数量与最高置信度 |
| ③ 不同场景 / 拍摄角度 | `results/evidence_level4/08_情况③不同场景与角度.jpg` | 三张不同拍摄时段的代表图（20260727 10:41 / 14:30 / 20260728 10:18） |
| 整体效果图 | `results/evidence_level4/03_检测效果图_拼接.jpg` | 3×3 共 9 格汇总，可直接用于报告 |

> ①的素材取自 `dataset/images/val`（不在 train 集内），但它们参与过验证，
> 因此只作展示、不计入 6.3 的严格评测指标；6.3 的指标只来自 6 张移出数据集的图片。
> ③的代表图按文件名排序取首/中/末三张，覆盖不同拍摄时段，选法可复现。

### 6.6 验证标准逐条对照

执行 `python check_level4.py`，输出 `results/evidence_level4/09_Level4验收对照表.txt`，
把考核给出的 5 条验证标准逐条自动判定，结果 **5 / 5 通过**：

| # | 考核验证标准 | 结论 | 主要证据 |
| --- | --- | --- | --- |
| 1 | 正确加载自己训练得到的模型 | 通过 | `01_加载模型.txt`（8 项 PASS） |
| 2 | 对若干张训练集之外的新图片完成推理 | 通过 | `02_对新图片推理.txt`、`test_images/_manifest.txt` |
| 3 | 保存目标检测结果 | 通过 | `04_保存检测结果.txt`、`05_检测结果汇总表.csv`、`results/saved_results/` |
| 4 | 提交至少一组清晰的最终检测效果图 | 通过 | `03`、`06`、`07`、`08` 四张效果图 |
| 5 | 结果中能正常显示目标框、类别名称和置信度 | 通过 | 画框图 + 6 列标签 + 按类别名分目录的裁剪图 |

> 截图 12：加载自己训练的模型（`01_加载模型.txt` 的结论段）
> 截图 13：单目标检测效果图（`06_情况①单目标检测.jpg`）
> 截图 14：多目标同时检测效果图（`07_情况②多目标检测.jpg`）
> 截图 15：不同场景 / 拍摄角度效果图（`08_情况③不同场景与角度.jpg`）
> 截图 16：终端输出的类别与置信度（`02_对新图片推理.txt`）
> 截图 17：保存结果与一致性校验通过（`04_保存检测结果.txt` 的 [四] 段）
> 截图 18：整体拼接效果图（`03_检测效果图_拼接.jpg`）
> 截图 19：Level 4 验收对照表（`09_Level4验收对照表.txt`）

> 一键复现：`load_model.py` → `infer_and_check.py` → `save_detection_results.py` → `check_level4.py`

---

## 七、遇到的问题与解决（真实过程记录）

1. **PATH 中的 python 是“应用执行别名”**：直接敲 `python` 报“系统找不到指定的路径”。
   解决：改用 `py -3.14` 或虚拟环境中的解释器全路径。
2. **LabelImg 1.8.6 与新版 PyQt5 不兼容，反复闪退**。
   现象：按 `W` 画框时进程直接消失，退出码 `0xC0000409`，终端没有任何报错
   （PyQt5 在 Qt 槽函数中遇到未捕获异常会直接终止进程，因此看不到 Python 的 traceback）。
   排查：实测 PyQt5 5.15.6 / 5.15.9 + sip 12.11~12.13 **均拒收 float 参数**
   （`setValue()`、`QPainter.drawRect / drawLine / drawText`），5.15.2 / 5.15.4 无对应轮子，
   即“免补丁即可用”的版本组合不存在。
   解决：写了 6 个幂等补丁脚本（`_patch_labelimg.py`、`_fix_labelimg_painter.py`、
   `_fix_labelimg_w_key.py`、`_fix_labelimg_asserts.py`、`_fix_libs_asserts.py`、
   `_fix_classes_txt.py`），把传给 Qt 的浮点参数显式转 `int()`、把热路径上的 assert 改为优雅处理；
   并另建 Python 3.11.9 专用标注环境 `C:\Users\dell\label-env` 作为可切换的备用环境
   （补丁脚本 `_finish_label_env.py` 会在该环境重复打一遍）。
   同时保留了零依赖的备用标注工具 `label_tool.py`（标准库 http.server + Canvas 前端），
   在补丁打通的空档期用于不中断标注进度。
3. **项目路径含中文与空格**（`D:\研一\git 练习`）导致部分工具出现权限与路径异常。
   解决：虚拟环境安装到 ASCII 路径 `C:\Users\dell\yolo-env`；所有脚本内部统一用
   `Path(__file__).resolve().parent` 定位，避免依赖当前工作目录。
4. **4GB 显存上限**：CUDA 预标注时出现 `CUDA out of memory`、`device not ready`（显卡被其它程序占用）。
   解决：预标注改用 CPU 推理（`--device cpu`），训练阶段 batch 降到 8（必要时 4）。
5. **没有标签的图片会被当作“背景图”参与训练**：949 张里若大量图片无标签，模型会被教成
   “这些图里什么都没有”，缺失类别永远学不到。
   解决：在 `train.py` 中加入训练前预检 —— 统计每类实例数，缺类别或背景图过多时直接中止并提示，
   可用 `--subset`（只训练已标注图片）先验证流程。
6. **两次划分脚本结果不一致**：早期用 PowerShell 写的 `split_dataset.ps1` 与项目最终的
   `split_dataset.py` 随机过程不同，同一批图片会落到不同 split（见 3.2 节）。
   解决：统一以 `split_dataset.py`（seed=42）为唯一划分入口，重新生成数据集，并把已经标好的
   219 个标签按图片实际所在的 split 重新归位，做到“换划分不丢标注”。
7. **预标注召回有限**（过程记录）：最初的 COCO 预标注在 football 上漏检严重，早期仅 33 张 football 完成标注。
   解决：改为用第一版模型做迭代预标注 + 人工复核，最终三类实例数达到
   obstacle 121 / cola 311 / football 777（合计 1209 个标注框）。
   并用 `audit_labels.py` 的“来源目录先验”自动找出疑似漏标。
   **诚实说明**：标注方式为「模型预标注 + 人工复核」（`auto_annotate_rest.py` 生成候选框、
   `refine_auto_labels.py` 按类别阈值筛选与消解冲突），**不是全部逐框手工绘制**，报告与提交中均如实标注。

---

## 八、各 Level 完成情况

| Level | 内容 | 状态 | 验证材料 |
| --- | --- | --- | --- |
| Level 1 | 环境配置与数据集整理 | 已完成 | 截图 1~4 |
| Level 2 | 目标标注 | 已完成 | 截图 5~7 |
| Level 3 | 模型训练（§1 配置数据集 / §2 启动训练 / §3 查看结果） | 已完成 | 截图 7.1、8~11 |
| Level 4 | 新图片目标检测（**及格线**） | 已完成 | 截图 12~19 |
| Level 5 | 本地识别页面（进阶，加分项） | 未开始 | — |

Level 4 的 5 条验证标准由 `check_level4.py` 自动核对，结果 **5 / 5 通过**（详见 6.6 节）。
一键复现命令：`load_model.py` → `infer_and_check.py` → `save_detection_results.py` → `check_level4.py`。

---

## 九、总结与不足

- 完整走通了“环境配置 → 数据整理 → 标注 → 训练 → 推理 → 验收”的目标检测工程链路；
  Level 4 的 5 条验证标准由 `check_level4.py` 自动核对，结果 **5 / 5 通过**（详见 6.6 节）。
- 标注方面：COCO 预标注只能覆盖约 23% 的图片（`obstacle` 类在 COCO 中没有对应类别，只能人工标）。
  后续用第一版模型做主动学习式的二次预标注（`auto_annotate_rest.py` + `refine_auto_labels.py` 按类别阈值筛选、
  消解跨类冲突）并逐张人工复核，最终三类实例数达到 **obstacle 121 / cola 311 / football 777
  （合计 1209 个标注框）**。
  **诚实说明**：标注方式为「模型预标注 + 人工复核」，并非全部逐框手工绘制，报告与提交中均如实标注。
- 仍存在的不足：
  1. **极小目标召回不足**：第 4 关 6 张新图片上的 4 个漏检、2 个误报全部集中在框面积 <0.001
     （小于整图千分之一）的目标上，而所有命中目标的面积都 ≥0.0025；
  2. **`obstacle` 类指标偏低**：以 `runs/detect/train` 的 `best.pt` 在当前验证集上评测，
     `obstacle` 的 mAP50 仅 0.630（召回 0.549），明显低于 `cola` 0.961 与 `football` 0.866。
     原因是“障碍物”语义宽泛、类内差异大且样本量最少（121 个框）；
     此外“橙色球 / 橙色方块”究竟算 `obstacle` 还是 `football` 这一标注边界问题，
     需要全局统一后重新训练才能改善；
  3. **标注来源不完全统一**：部分标签由模型预标注生成，与纯人工标注的标签在框的松紧度上存在细微差异；
  4. **验证集偏小**：188 张验证图片、239 个实例，单轮指标抖动较大，指标只能作为趋势参考。
- 后续计划：统一 `obstacle` 的标注边界后补标并重训；用更高分辨率的训练尺度或切片推理（SAHI 思路）
  改善极小目标；尝试用 Streamlit / Gradio 完成 Level 5 本地识别页面。
