# 基于YOLO的机器人场景目标检测 工程实践报告
姓名：张子煦
学号：202611160502
专业：计算机科学与技术
## 项目概述
本项目为机器人实验室2026工程实践能力考核项目，基于YOLOv8完成机器人场景目标检测。
任务目标：识别三类目标：`0: obstacle（障碍物）`、`1: cola（可乐罐）`、`2: football（足球）`。
完整流程：开发环境搭建 → 数据集整理与划分 → 图片标注 → 模型训练 → 新图片推理测试。
项目最低要求完成Level4，本项目完成Level4（可选：进阶完成Level5）。

## Level1：准备YOLO开发环境与整理数据集
### 1.1 环境配置
- Python版本：3.10.22
- 框架：PyTorch + Ultralytics YOLO
- 硬件：笔记本 RTX 5070Ti
- CUDA状态：可用

**验证截图1：Python版本验证终端截图**
![[level1_python_version.png.png]]

**验证截图2：PyTorch+GPU可用性验证终端截图**
![[level1_pytorch_gpu.png.png]]

**验证截图3：yolo安装验证终端截图**
![[level1_yolo_checks.png.png]]

**验证截图4：yolo安装验证终端截图
![[level1_yolo_checks.png (2).png]]




### 1.2 原始数据集检查
原始数据共3个文件夹：obstacle、cola、football。
统计：
- obstacle图片数量：316 张
- cola图片数量：293 张
- football图片数量：340 张
**检查结果：图片均可正常读取，无损坏文件。**


### 1.3 数据集目录搭建与划分
采用YOLO标准数据集目录结构，训练集:验证集 = 8:2。
> 划分策略：对三类目标图片分别按比例随机划分，保证训练、验证集中都包含全部类别，避免类别分布不均衡。

**目录结构截图2：项目文件夹树截图**
![[folder_tree.png.png]]

## Level2：完成目标检测标注
### 2.1 标注工具
使用LabelImg工具进行标注，标签输出格式选择YOLO。
类别定义：
|类别编号|类别名称|中文含义|
| ---- | ---- | ---- |
|0|obstacle|障碍物|
|1|cola|可乐罐|
|2|football|足球|

### 2.2 标注流程
1. 加载图片，对目标物体绘制矩形框
2. 选择对应类别编号
3. 保存，自动生成同名`.txt`标签文件
> YOLO标签格式说明：`class x_center y_center width height`，坐标全部经过归一化，取值范围0~1。

**截图3：LabelImg标注工具运行截图**
![[level2_labeltool.png.png]]

**截图4：标注完成图片可视化截图**
![[level2_labeltool.png]]

**截图5：对应txt标签文件内容截图**
![[level2_txt.png.png]]

## Level3：完成YOLO模型训练
### 3.1 数据集配置文件 data.yaml
yaml
path: ../dataset
train: images/train
val: images/val
names:
  0: obstacle
  1: cola
  2: football

### 3.2 训练参数
- 模型：YOLOv8s
- epoch：50
- batch：8
- 设备：GPU RTX5070Ti

### 3.3 训练过程与结果
训练启动成功，终端输出训练日志。训练完成后，在 `runs/train/weights/` 生成模型权重`best.pt`、`last.pt`。

**截图 6：模型开始训练的终端日志截图**
![[screenshot6_train_log.png.png]]
![[screenshot6_train_log.png (2).png]]
**截图 7：训练结果目录截图**
![[screenshot7_train_dir.png.png]]

**截图 8：训练指标图（Loss、Precision、Recall）**
![[screenshot8_results_curve.png.png]]

指标简要说明：
- Loss：损失函数，数值越低代表模型预测误差越小
- Precision（精确率）：预测出来的目标里真正目标的占比
- Recall（召回率）：图片里真实目标被模型成功检出的比例

## Level4：完成新图片目标检测
使用训练得到的`best.pt`权重，对**训练集以外**的新图片做推理测试。

测试场景：
1. 单目标检测
2. 多目标同图检测
3. 不同角度场景检测

**截图 9：新图片推理检测效果图（至少一张）**
![[screenshot9_detect.png.png]]

结果说明：图中成功画出目标框，显示类别名称与置信度，模型可以正确识别 obstacle、cola、football 三类物体。


## 遇到的问题及解决方案

###  问题 1：CPU 训练速度较慢

**原因**：使用 CPU 训练，每轮约 50-58 秒，训练耗时较长。

**解决方案**：改用 GPU 版 PyTorch，训练 50 轮仅需约 12 分钟，顺利完成项目。
### 问题 2：LabelImg 按 W 键画框闪退

**原因**：PyQt5 版本与 Python 版本不匹配，或配置缓存文件损坏。

**解决方案**：
1. 降级 PyQt5 到 5.15.10
2. 删除 `.labelImgSettings.pkl` 缓存文件
3. 使用鼠标点击菜单 Edit → Create RectBox 代替 W 快捷键

### 问题 3：GitHub 下载慢

**原因**：网络访问 GitHub 不稳定。

**解决方案**：使用 Watt Toolkit 或 ghfast.top 镜像加速下载。

### 问题 4：obstacle 类别检测效果差

**原因**：自动预标注脚本只覆盖了 COCO 认识的 bottle/cup 和 sports ball，obstacle 是自定义类别，预训练模型不认识。

**解决方案**：后续需用 LabelImg 手动给障碍物画框，补充 obstacle 标注数据后重新训练。
### 问题 5：LabelImg 标注格式选错，生成的是 xml 不是 txt

**原因**：LabelImg 默认保存格式是 PascalVOC（xml），不是 YOLO 格式 每次打开 LabelImg 都要手动切换左上角格式下拉框

**解决方案**：标注前先把格式切换为 YOLO，生成 .txt 文件；并确认每张图片都有同名 txt。


## 总结
本次工程实践完整走完 YOLO 目标检测工程流程，从环境配置、数据集整理标注、模型训练到推理。
掌握了数据集划分、YOLO 标签格式、模型训练与推理的工程使用方法，理解目标检测基础指标。
学会查阅官方文档、排查报错，提升工程实践能力，为后续机器人视觉方向打下基础。

## 项目文件清单
- train.py：模型训练脚本
- predict.py：图片推理脚本
- data.yaml：数据集配置
- requirements.txt：项目依赖
- dataset：数据集图片与标签
- runs：训练输出结果与权重
- report.md：项目报告
