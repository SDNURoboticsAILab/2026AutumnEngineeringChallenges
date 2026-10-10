
# 基于YOLO的机器人场景目标检测实验报告

## 一、 项目背景与目标
本项目旨在针对机器人场景中的特定物体（足球、天蓝色正方体、纯黑色无标签大饮料瓶）进行目标检测，为后续机器人的抓取或环境感知提供视觉基础。

## 二、 环境配置与准备
*   **硬件**：NVIDIA GeForce GTX 1660 Ti
*   **软件**：Anaconda + PyCharm + CUDA 11.8
*   **环境验证截图**：![环境验证截图](resullts/26.10.8安装终端截图1.PNG)
    

## 三、 数据集准备与标注
1. **数据采集**：共收集指定场景图片1000张，包含不同角度和光照。
2. **数据标注**：使用 LabelImg 工具进行标注，保存为 YOLO 格式（生成同名 `.txt` 文件），包含三个类别（类别编号 0-cola, 1-football, 2-obstacle）。
3. **数据集划分**：按照 7:3 的比例，随机划分为训练集（665张）和验证集（284张）。
4. **配置文件（data.yaml）**：
```yaml
path: D:\deeplearning\yolo_project\dataset # dataset root dir
train: images/train
val: images/val


# Classes
names:
  0: cola
  1: football
  2: obstacle
```

## 四、 模型训练过程
*   **训练脚本（train.py）**：
    （在此处插入你的 `train.py` 代码截图或代码块）
    *注：设置了 batch=-1（自动调节显存）、cache="ram"（内存缓存加速）等参数。*
*   **开始训练**：
    ![训练截图](resullts/26.10.5训练运行截图.PNG)
*   **训练参数记录**：
    见 `runs/detect/train/args.yaml`。

## 五、 实验结果与分析
![results](resullts/results.PNG)
**结果分析**：
从上图可以看出，随着训练轮次的增加，`box_loss`、`cls_loss`、`dfl_loss` 均呈平稳下降趋势。最终在验证集上，`mAP50` 达到了 0.99，说明模型在检测这三个特定物体时表现极为优秀。

## 六、 模型验证与测试
1. **模型权重文件**：训练完成后最优模型保存在 `runs/detect/train/weights/best.pt`。
2. **新图片推理**：
   ![数据集外推理](resullts/数据集外推理2.jpg)

## 七、 问题与解决过程记录
*   **1.**：在新建 PyCharm 项目后，环境配置报错 `lateinit property envs_dirs has not been initialized`。
    *   **解决**：通过 Conda 初始化命令 `conda init`，并在 PyCharm 里选择根目录下的 `_conda.exe` 作为解释器解决。
*   **2.**：把训练脚本换到新文件夹后，报错找不到 `ultralytics` 模块和 `obsr.yaml`。
    *   **解决**：将 PyCharm 的 Python 解释器切换到正确的虚拟环境，并将数据路径改为绝对路径解决。
*   **3.**：GitHub 网页端提交时报错 `File could not be edited`。
    *   **解决**：通过刷新页面或使用 `Create a new branch` 选项绕过网页卡顿，后续改用本地整理文件直接拖拽上传。
*   **4.**：找不到合适的训练集外图片用于目标检测。
    *   **解决**：购买材料，自行拍摄图片。

## 八、 总结与展望
通过本次工程实践，完全是小白的我掌握了从数据标注、YOLO模型训练、指标分析到推理验证的全流程。希望能和学长学姐们多多学习，提升自己。
