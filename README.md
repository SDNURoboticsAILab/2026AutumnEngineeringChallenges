<<<<<<< HEAD
# 机器人实验室 2026 工程实践能力考核说明

欢迎你，未来的技术大佬！

为了更全面地评估大家的工程素养、学习能力与动手实践能力，机器人实验室继续设立工程实践能力考核通道。

我们相信，扎实的理论知识固然重要，但能够主动学习新知识、阅读文档、配置环境、分析问题并将想法真正落地，同样是工程师非常重要的能力。

完成本次考核并按照要求提交成果的同学，将获得 **免除笔试环节，直通最终面试** 的机会！

本次考核主要面向刚接触人工智能、机器人、嵌入式开发等方向的新生。大家不需要在开始之前就掌握所有相关知识。本次考核更关注你的 **学习能力、探索精神、工程实践能力以及解决问题的过程**。

我们鼓励大家积极查阅资料，如 [CSDN](https://www.csdn.net/)、[Bilibili](https://www.bilibili.com/)、[GitHub](https://github.com/)、官方文档等，也可以合理使用 ChatGPT、Qwen、DeepSeek、Gemini 等 AI 工具辅助学习和排查问题。但最终提交的代码、实验过程和项目报告应由本人实际完成，并能够理解和解释。

最有价值的不只是最终运行成功的程序，而是你在探索过程中留下的思考与实践记录。因此，**学习过程、遇到的问题、解决方案以及个人总结**，都将是本次考核的重要参考。

---

## 一、考核项目选择（三选一）

本次工程实践提供三个不同方向的挑战项目。请根据自己的兴趣与现有基础，任选其一完成。

### 项目一：基于 YOLO 的机器人场景目标检测

- **项目简介：** 使用实验室提供的 `obstacle`、`cola`、`football` 三类图片数据，自行完成数据整理、目标标注、YOLO 模型训练，并使用自己训练得到的模型对新图片进行目标检测。
- **核心技术栈：** Python、Linux、PyTorch、YOLO、目标检测、数据集标注与处理。
- **考核及格线：** 至少完成 **Level 4**，能够使用自己训练得到的模型对新图片进行检测，并展示目标框、类别和置信度。
- **发展方向：** 适合对人工智能、计算机视觉、深度学习、机器人视觉等方向感兴趣的同学。
- **详细要求：** 请查看项目一目录下的 `README.md`。

### 项目二：四足机器人仿真环境搭建

- **项目简介：** 在 Linux 环境下使用 Docker 搭建 CyberDog 四足机器人仿真环境，运行 Gazebo 与 RViz2，并进一步了解 ROS2、机器人传感器以及运动控制等基础内容。
- **核心技术栈：** Linux、Docker、ROS2、Gazebo、RViz2、URDF/Xacro、机器人传感器与通信机制。
- **考核及格线：** 至少完成 **Level 3**，成功运行 Gazebo + RViz2 四足机器人仿真程序。
- **发展方向：** 适合对机器人、ROS2、具身智能、运动控制、自主导航等方向感兴趣的同学。
- **详细要求：** 请查看项目二目录下的 `README.md`。

### 项目三：基于树莓派和 OpenCV 的人脸识别门禁系统

- **项目简介：** 使用树莓派、摄像头和 OpenCV 搭建基础人脸识别系统，从系统烧录、远程连接、摄像头调用逐步实现人脸数据采集、训练和识别。
- **核心技术栈：** Raspberry Pi、Linux、Python、OpenCV、计算机视觉、嵌入式开发、硬件交互。
- **考核及格线：** 至少完成 **Level 3**，能够对静态图片中的人物进行人脸识别。
- **发展方向：** 适合对嵌入式开发、物联网、智能硬件、计算机视觉以及软硬件结合感兴趣的同学。
- **详细要求：** 请查看项目三目录下的 `README.md`。

---

## 二、评估标准

我们将从以下几个方面对项目成果进行综合评估：

1. **项目完成度：** 是否达到或超过对应项目规定的最低考核等级。
2. **过程文档记录：** 是否记录学习过程、环境配置、调试过程、遇到的问题及解决方案。
3. **代码与工程质量：** 项目结构是否清晰，代码是否易于理解，配置和说明是否完整。
4. **实验结果：** 是否能够证明项目确实运行成功，并提供必要的截图、日志或演示视频。
5. **创新与扩展：** 是否在完成基础要求后继续挑战更高等级或进行了额外功能扩展。
6. **独立思考与原创性：** 是否能够体现自己的理解、实践和总结，而不是简单复制他人的项目或 AI 输出。

---

## 三、提交要求

请将所有相关成果整理后，在代码仓库中对应题目目录下创建一个以自己 **学号** 命名的文件夹，并提交 Pull Request（PR）。

提交方式与往年保持一致。

提交内容至少包括：

- **完整的项目源代码或配置文件：** 包含完成项目所需的主要代码、脚本及配置文件。
- **详细的项目报告（Markdown 格式）：** 记录学习过程、环境配置、实现过程、遇到的问题、解决方案和个人总结。
- **最终成果展示：** 提供清晰的运行截图、实验结果或简短演示视频。
- **提交格式：** Git Commit Message 与 Pull Request 标题应符合 [Conventional Commits（约定式提交）](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 规范。

Pull Request 标题请注明 **专业 + 班级 + 姓名**。

例如：

```text
feat(YOLO): 计工本XXXX张三提交项目一成果
```

```text
feat(SIM): 计工本XXXX张三提交项目二成果
```

```text
feat(RasPi): 计工本XXXX张三提交项目三成果
```

第一次使用 Git 与 GitHub 的同学，可以提前了解：

- [Git 官网](https://git-scm.com/)
- [GitHub 官网](https://github.com/)
- [GitHub Docs：开始使用 GitHub](https://docs.github.com/zh/get-started)
- [GitHub Docs：Pull Request](https://docs.github.com/zh/pull-requests)
=======
# 基于 YOLO 的机器人场景目标检测

- **项目简介**：本项目围绕 2026 年全国大学生计算机系统能力大赛智能系统创新设计赛（小米杯）相关机器人场景展开。你将使用实验室提供的 `obstacle`、`cola`、`football` 三类图片数据，自行完成数据整理、目标标注、YOLO 模型训练，并使用自己训练得到的模型对新图片进行目标检测。

  通过本项目，你将完整经历一次目标检测项目的基本流程，包括 **数据集整理、目标标注、模型训练、模型评估与推理应用**。项目重点不在于从零实现 YOLO 算法，而在于掌握如何使用成熟的开源工具完成一个完整的计算机视觉工程实践。

- **核心技术栈**：Python、Linux、PyTorch、YOLO、目标检测、数据集标注与处理。

- **考核及格线**：至少完成 **Level 4**（使用自己训练得到的模型，对新的图片完成目标检测）。

- **发展方向**：适合对人工智能、计算机视觉、深度学习、机器人视觉与模型训练感兴趣的同学。

---

## 一、考核目的与总提交要求

本考核旨在检验参与者对目标检测项目完整流程的基本掌握情况，包括开发环境配置、数据集整理、数据标注、模型训练、推理测试以及实验结果整理等工程实践能力。

### 1. 参赛者需要完成以下任务

1. 配置 Python 与 YOLO 开发环境；
2. 整理实验室提供的 `obstacle`、`cola`、`football` 三类图片数据；
3. 完成目标检测数据标注，并转换为 YOLO 数据集格式；
4. 使用整理后的数据集训练 YOLO 模型；
5. 使用自己训练得到的模型对新的图片进行目标检测；
6. 保存并展示模型训练与推理结果。

```text
obstacle 数据集
cola 数据集
football 数据集
```

> 注意：以上三个文件夹中的图片为本次任务提供的原始数据。你需要自行完成数据整理，并按照 YOLO 数据集格式组织训练集、验证集与对应标签。

### 2. 总提交要求

1. **项目报告**：详细记录环境配置、数据集整理、目标标注、模型训练、推理测试、遇到的问题及解决方案，并包含各 Level 的验证截图；
2. **完整项目代码与配置文件**：包括训练脚本、推理脚本、数据集配置文件及必要说明；
3. **最终实验结果**：至少包含模型训练结果截图与新图片目标检测结果；完成 Level 5 的同学还应提交本地页面运行截图或演示视频；
4. **项目提交**：在指定 GitHub 仓库对应目录中，以本人学号建立文件夹并提交 Pull Request（PR）。

---

## 二、考核步骤

## Level 1：准备 YOLO 开发环境与整理数据集

在正式训练模型前，需要先完成开发环境配置，并将原始图片整理为适合后续标注与训练的数据集结构。

### 实践步骤

#### 1. 配置开发环境

根据自己的实际情况选择合适的开发环境，并完成 Python、PyTorch 与 YOLO 的安装。

推荐使用当前主流的 YOLO 实现，例如 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)。

参考资料：

- [Ultralytics YOLO 官方文档](https://docs.ultralytics.com/)
- [Ultralytics YOLO GitHub](https://github.com/ultralytics/ultralytics)

> **什么是 YOLO？**
>
> YOLO（You Only Look Once）是一类常用的目标检测模型，可以在一张图片中同时判断“图片里有什么目标”以及“目标位于什么位置”。

#### 2. 检查原始数据

检查实验室提供的三组图片：

```text
obstacle
cola
football
```

了解图片数量、图片格式、分辨率以及是否存在无法正常读取的图片。

#### 3. 建立 YOLO 数据集目录

按照 YOLO 数据集格式建立项目目录，并将数据划分为训练集和验证集。

推荐目录结构：

```text
yolo_project/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── runs/
├── train.py
├── predict.py
└── data.yaml
```

参考资料：

- [Ultralytics：Train Mode](https://docs.ultralytics.com/modes/train/)
- [Ultralytics：数据集格式说明](https://docs.ultralytics.com/datasets/)

### 验证标准与提交内容

1. 提交 Python、PyTorch、Ultralytics YOLO 成功安装的终端截图；
2. 提交整理后的项目目录结构截图；
3. 在项目报告中说明训练集与验证集的划分方式。

---

## Level 2：完成目标检测标注

目标检测模型除了需要图片，还需要对应的目标位置与类别信息，因此需要对图片中的目标进行标注。

### 实践步骤

#### 1. 选择标注工具

可以使用以下工具：

- [LabelImg GitHub](https://github.com/HumanSignal/labelImg)
- [Roboflow](https://roboflow.com/)

> **什么是“标注”？**
>
> 标注就是在图片中把目标框出来，并告诉模型“这个框里是什么”。例如，在一张足球图片中圈出足球，并标记类别 `football`。

#### 2. 完成三类目标标注

本项目至少包含以下三个类别：

```text
0 obstacle
1 cola
2 football
```

每张参与训练的图片都应具有对应的标签文件。

#### 3. 了解 YOLO 标签格式

你需要能够基本理解：

- 什么是目标检测；
- 什么是类别；
- 什么是边界框（Bounding Box）；
- YOLO 标签文件中的数字分别表示什么。

参考资料：

- [Ultralytics 数据集格式文档](https://docs.ultralytics.com/datasets/)

> 不要求死记公式，但需要能够看懂一行 YOLO 标签数据的大致含义。

### 验证标准与提交内容

1. 提交标注工具运行截图；
2. 提交至少一张已经完成目标框标注的图片截图；
3. 提交对应 YOLO 标签文件内容截图；
4. 在项目报告中说明三个类别的编号及其含义。

---

## Level 3：完成 YOLO 模型训练

完成数据整理与标注后，使用自己的数据集训练 YOLO 模型。

### 实践步骤

#### 1. 配置数据集文件

编写 `data.yaml`，正确设置训练集、验证集路径以及类别信息。

例如：

```yaml
names:
  0: obstacle
  1: cola
  2: football
```

#### 2. 启动模型训练

使用 Ultralytics YOLO 或其他成熟 YOLO 实现完成训练。

参考资料：

- [Ultralytics：Train Mode](https://docs.ultralytics.com/modes/train/)
- [Ultralytics：Python 使用方式](https://docs.ultralytics.com/usage/python/)

> **什么是“训练模型”？**
>
> 训练就是把已经标注好的图片交给模型，让模型不断学习图片中的目标特征和位置，最终得到一个可以用于目标检测的新模型。
>
> 本项目不要求从零实现 YOLO。能够正确使用成熟的开源实现完成训练，就是本项目希望你掌握的重要工程能力。

#### 3. 查看训练结果

至少需要能够查看训练过程中的：

- Loss；
- Precision；
- Recall；
- 训练结果文件；
- 模型权重文件。

相关指标参考：

- [Ultralytics：Metrics 指标说明](https://docs.ultralytics.com/guides/yolo-performance-metrics/)

### 验证标准与提交内容

1. 提交模型正常开始训练的终端截图；
2. 提交训练完成后的结果目录截图；
3. 提交 Loss、Precision、Recall 等训练结果图；
4. 提交训练得到的模型权重文件信息；
5. 使用训练得到的模型完成一次验证或测试。

达到这一等级后，你应该已经拥有一个使用自己数据集训练得到的目标检测模型，而不是只调用别人已经训练好的模型。

---

## Level 4：完成新图片目标检测

**本项目及格线要求：至少完成 Level 4。**

使用自己训练得到的模型，对训练集之外的新图片进行目标检测。

### 实践步骤

#### 1. 加载训练模型

加载自己训练得到的模型权重。

#### 2. 对新图片进行推理

选择训练集之外的新图片进行目标检测。

参考资料：

- [Ultralytics：Predict Mode](https://docs.ultralytics.com/modes/predict/)

> **什么是“推理（Inference）”？**
>
> 推理不是再次训练，而是把模型没有见过的新图片交给已经训练好的模型，让模型根据训练过程中学到的内容判断图片中有哪些目标。

#### 3. 保存检测结果

检测结果应能够显示：

- 目标框；
- 类别名称；
- 置信度。

建议至少展示以下几种情况：

```text
① 单个目标检测
② 多个目标同时检测
③ 不同场景或不同拍摄角度下的检测
```

### 验证标准与提交内容

1. 正确加载自己训练得到的模型；
2. 对若干张训练集之外的新图片完成推理；
3. 保存目标检测结果；
4. 提交至少一组清晰的最终检测效果图；
5. 检测结果中能够正常显示目标框、类别名称和置信度。

---


## Level 5：制作本地图片识别页面

在完成模型训练和新图片推理后，可以进一步将模型封装成一个简单的本地可视化应用。

本 Level 作为进阶挑战，不影响本项目 **Level 4** 的及格要求。

### 实践步骤

#### 1. 搭建本地前端页面

选择一种自己熟悉或感兴趣的方式制作简单的本地页面，例如：

- [Streamlit](https://streamlit.io/)
- [Gradio](https://www.gradio.app/)
- Flask + HTML
- 其他能够在本地运行的前端或 Web 框架

不要求页面设计复杂，能够完成基本交互即可。

#### 2. 实现图片上传功能

页面需要允许用户从本地电脑选择并上传一张图片。

基本流程可以理解为：

```text
打开本地页面
    ↓
选择并上传图片
    ↓
后端加载自己训练的 YOLO 模型
    ↓
对上传图片进行目标检测
    ↓
在页面中显示检测结果
```

#### 3. 调用自己训练的 YOLO 模型

页面后端需要加载自己在 Level 3 中训练得到的模型，而不是直接使用未经本项目数据训练的通用模型。

上传图片后，程序应自动完成推理。

#### 4. 在页面中展示检测结果

页面中至少需要展示：

- 用户上传的原始图片；
- YOLO 检测后的结果图片；
- 识别出的目标类别；
- 对应的置信度信息。

页面样式不作为主要考核内容，重点是能够完成从 **图片上传 → 模型推理 → 结果展示** 的完整流程。

### 验证标准与提交内容

1. 能够在本地电脑成功启动前端页面；
2. 页面能够正常上传图片；
3. 能够调用自己训练得到的 YOLO 模型完成目标检测；
4. 页面能够正确显示检测后的图片、目标类别和置信度；
5. 提交页面运行截图或简短演示视频；
6. 提交实现该页面所使用的源代码及必要依赖说明。


## 三、提交规范

请将所有相关成果整理后，在指定代码仓库中创建一个以自己学号命名的文件夹，并提交 Pull Request（PR）。

建议目录结构：

```text
你的学号/
├── README.md
├── train.py
├── predict.py
├── data.yaml
├── requirements.txt
├── dataset/
├── runs/
├── results/
└── report.md
```

提交内容至少包括：

1. **完整项目代码**：包含模型训练、推理以及必要配置文件；
2. **项目报告**：使用 Markdown 编写，记录环境、数据处理、标注、训练、实验结果、问题与解决过程；
3. **最终成果展示**：至少包含模型训练结果截图和新图片检测结果截图；
4. **过程记录**：各 Level 的完成过程与验证截图。

Pull Request 标题建议使用：

```text
feat(YOLO): 班级+姓名提交工程实践项目
```

GitHub、Git 与 Markdown 相关资料：

- [GitHub 官网](https://github.com/)
- [GitHub Docs：开始使用 GitHub](https://docs.github.com/zh/get-started)
- [GitHub Docs：Pull Request](https://docs.github.com/zh/pull-requests)
- [Git 官方网站](https://git-scm.com/)
>>>>>>> 8e9e2af8ab4db53e61030ab3af9df90368030e25
- [Markdown 基本语法](https://markdown.com.cn/basic-syntax/)

---

<<<<<<< HEAD
## 四、参考资料和可能有用的工具

### GitHub 等国外网站访问

- [Watt Toolkit](https://steampp.net/)

### Markdown

- [Markdown 基本语法](https://markdown.com.cn/basic-syntax/)

### Git

- [Git 官网](https://git-scm.com/)
- [GitHub Docs](https://docs.github.com/zh)

### 大语言模型

可以合理使用 AI 工具辅助学习、理解代码和分析报错，例如：

#### 国内

- [Qwen](https://chat.qwen.ai/)
- [Kimi](https://www.kimi.com/)
- [DeepSeek](https://chat.deepseek.com/)
- [豆包](https://www.doubao.com/chat/)

#### 国际

- [ChatGPT](https://chatgpt.com/)
- [Claude](https://claude.ai/)
- [Gemini](https://gemini.google.com/)
- [Google AI Studio](https://aistudio.google.com/)

---

最后，希望本次工程实践挑战能够成为大家接触人工智能、机器人与嵌入式开发的一次真正实践。

请记住，**过程比结果更重要**。

不会没有关系，重要的是遇到问题后能够主动查资料、尝试解决，并把这个过程真正记录下来。

期待看到你们的作品！

祝各位挑战顺利！

**山东师范大学机器人实验室**

**2026 年**
=======
## 四、参考与推荐学习资源

如果你想进一步学习 YOLO、计算机视觉与机器人视觉，推荐参考以下资料：

- [Ultralytics YOLO GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics YOLO 官方文档](https://docs.ultralytics.com/)
- [Ultralytics 数据集文档](https://docs.ultralytics.com/datasets/)
- [Ultralytics Train Mode](https://docs.ultralytics.com/modes/train/)
- [Ultralytics Predict Mode](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics Metrics 指标说明](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [LabelImg GitHub](https://github.com/HumanSignal/labelImg)
- [Roboflow](https://roboflow.com/)

### 参考背景资料

2026 年全国大学生计算机系统能力大赛智能系统创新设计赛（小米杯）相关资料：

- [全国大学生计算机系统能力大赛官网](https://is.educg.net/)
- [2026 年比赛相关信息](https://course.educg.net/)

---

最后，希望本次工程实践能够让你真正接触一次完整的人工智能目标检测项目开发流程。

你不需要一开始就会 YOLO，也不需要一开始就懂深度学习。真正重要的是：遇到不会的内容，能够查资料、跑实验、看报错、改代码，再继续往下推进。

**从一堆图片，到一个真正能够“看懂”图片的模型，这就是本次工程实践希望你完成的事情。**
>>>>>>> 8e9e2af8ab4db53e61030ab3af9df90368030e25
