# 基于 YOLO 的机器人场景目标检测

学号：20262128003

本目录用于完成 `obstacle`、`cola`、`football` 三类目标的标注、训练、验证和新图片推理。

## 类别编号

```text
0 obstacle
1 cola
2 football
```

## 工作流程

1. 运行 `prepare_dataset.py`，检查并划分原始图片。
2. 使用标注工具为 `dataset/images/train` 和 `dataset/images/val` 中的图片生成同名 YOLO 标签。
3. 运行 `validate_labels.py` 检查标签。
4. 运行 `train.py` 训练模型。
5. 运行 `predict.py` 检测 `dataset/images/test` 中未参加训练的图片。
6. 运行 `streamlit run app.py` 启动本地图片识别页面（Level 5），在浏览器中上传图片并查看检测结果。

中断后续训用 `resume_train.py`（从 `runs/yolo11n_3classes/weights/last.pt` 恢复）。

在仓库外层根目录可直接双击对应脚本：`train-yolo.cmd`（训练）、`predict-yolo.cmd`（批量推理）、`web-yolo.cmd`（启动识别页面）、`check-yolo.cmd`（环境自检）。

## 标注策略

本项目采用**每张图标注画面主目标**的方式，最终 train 759 张 + val 95 张，共 **854 个目标框，每张图 1 个框**。

标注时使用 `predefined_classes.txt` 固定类别顺序，确保与 `data.yaml` 中的类别编号一致——若两处顺序不一致，标签中的编号含义会整体错位。

**测试集只用于最终推理，不参与训练、验证和调参。** `data.yaml` 中只声明了 `train` 与 `val`，没有 `test` 键，因此测试集的标签不会进入训练流程。测试集另有 47 张图片保留了标注（14 张 cola、33 张 football），仅用于留出集评估。

> **已知局限**：训练与验证标签只覆盖画面主目标。若某张图中还存在同属这三类的次要目标而未被标注，该区域会被当作背景参与训练，可能使模型对该类目标产生漏检。相关分析与实测数据见报告第六节。

## 复现步骤

本目录**不包含 `dataset/images/`**（949 张图片，约 108 MB）。原因是它们为仓库中原始 `obstacle/`、`cola/`、`football/` 三个目录图片的重命名副本，且划分过程完全确定性，可以重建，故不在仓库中重复存放。

重建方法（在 `20262128003` 目录内执行）：

```powershell
python prepare_dataset.py
```

`prepare_dataset.py` 使用固定随机种子 `20262128003`，并先按文件名排序再打乱，因此**在任何机器上重建都会得到完全相同的划分与文件名**，与仓库中已有的 `dataset/labels/` 一一对应。

重建后建议验证一次：

```powershell
python validate_labels.py
```

输出 `检查通过，共检查 854 个目标框。` 即表示图片与标签完全匹配。

## 环境

环境依赖与安装顺序见 `requirements.txt`，完整版本快照见仓库外层 `environment/requirements-lock.txt`。

关键点：PyTorch 必须从 CUDA 11.8 专用源安装（`--index-url https://download.pytorch.org/whl/cu118`），否则会装成 CPU 版本、无法使用 GPU。

## 标注工具

在仓库外层根目录双击 `label-train.cmd` 标注训练集，完成后双击 `label-val.cmd` 标注验证集。启动脚本已处理中文路径造成的 Qt 插件定位问题，并自动设置图片目录、类别文件和标签保存目录。
