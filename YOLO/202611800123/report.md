# 基于 YOLO 的机器人场景目标检测

## 1. 环境配置

- 操作系统：Windows 11
- Python：3.10
- PyTorch：2.14.1+cpu
- Ultralytics YOLO：8.4.174
- 是否使用 GPU：否（纯 CPU 训练）

安装验证截图：

![环境配置验证](images/env_conda.png)

![激活yolo环境](images/env_yolo.png)

![安装PyTorch](images/env_torch.png)

![安装X-AnyLabeling](images/env_xanylabeling.png)

## 2. 数据集整理

- 原始数据：obstacle、cola、football
- 训练集/验证集划分：使用提供的 dataset 目录（train/val 已划分）
- 标签类别：obstacle、cola、football

类别配置文件：

![类别文件](images/classes_txt.png)

数据配置文件：

![data.yaml](images/data_yaml.png)

## 3. 目标标注

- 工具：X-AnyLabeling
- 类别编号：
  - 0 obstacle
  - 1 cola
  - 2 football

标注界面截图：

![标注界面](images/label_interface.png)

YOLO 标签文件内容截图：

![标签文件](images/label_txt.png)

## 4. 模型训练

- 模型：YOLOv8n
- 训练参数：epochs=30, imgsz=416, batch=4, device=cpu
- 最终指标：mAP50 = 0.977，Precision = 0.91，Recall = 0.973

训练终端截图：

![训练终端](images/train_terminal.png)

训练过程 Loss、Precision、Recall 曲线：

![训练结果](images/results.png)

## 5. 新图片推理（Level 4 及格线）

使用自己训练得到的 `best.pt`，对训练集之外的新图片进行推理，检测结果如下：

单目标检测：

![预测1](results/predict/image_rgb_20260727_120127.jpg)

多目标检测：

![预测2](results/predict/image_rgb_20260727_120134.jpg)

不同角度/场景检测：

![预测3](results/predict/image_rgb_20260727_120143.jpg)

结果分析：模型对足球、可乐、障碍物三类目标识别准确，最高置信度达 0.99，在 CPU 上单张推理约 8 毫秒。

## 6. 遇到的问题与解决

1. **GPU 版本 PyTorch 下载太慢**
   尝试清华、阿里云等镜像仍然很慢，最终改用 CPU 版 PyTorch，训练 30 轮仅需约 7 分钟，顺利完成项目。

2. **labelImg 安装报错**
   在 `yolo` 环境下安装 labelImg 失败，改用 X-AnyLabeling 进行标注，界面友好且支持直接导出 YOLO 格式。

3. **训练时路径错误**
   最初 `data.yaml` 中路径写成 `D:/`，实际项目在 `C:/Users/lenovo/Desktop/`，修改为绝对路径后训练正常。

4. **模型保存路径较深**
   训练结果被自动保存到 `runs/detect/runs/train/exp` 下，推理时需注意使用正确路径加载 `best.pt`。
