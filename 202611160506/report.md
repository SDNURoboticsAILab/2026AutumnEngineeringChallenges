\# 实验报告



\## 1. 环境说明

Windows 系统，AMD Ryzen 7 处理器，无 NVIDIA 显卡（CUDA不可用）。Python 3.11，ultralytics 8.4.174。



\## 2. 数据处理

使用 MakeSense 对60张图片进行了标注。排除了类别编号冲突（即不同文件夹独立标0的问题），手动将 cola 设为0，football 设为1，obstacle 设为2，并配置 data.yaml 完成加载。



\## 3. 训练过程

使用 yolov8n.pt 预训练权重，由于是 CPU 训练，参数设为 imgsz=320, batch=4, device=cpu，共训练了50个 epoch。



\## 4. 实验结果

模型在验证集上 mAP50 达到了 0.685，结果图和混淆矩阵保存在 runs 文件夹中。



\## 5. 新图片推理 (Level 4)

对训练集之外的新图片进行了推理，检测结果保存在 results 文件夹中。



\## 6. 问题与解决过程

遇到的最主要问题是：YOLO 报错找不到 txt 文件，且预测时名称与实物不对应。

解决方法：

1\. 文件夹拼写错误：将 `lables` 文件夹重命名为 `labels`。

2\. 标签类别冲突：手动修改 txt 第一列的类别编号，确保与 data.yaml 里的 names 顺序对应。

3\. CUDA不可用：训练命令中加上 `device=cpu` 并降低图片尺寸。

