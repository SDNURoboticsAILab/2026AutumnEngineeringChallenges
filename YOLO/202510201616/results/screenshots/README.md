# screenshots

本目录存放报告和提交要求所需的运行截图/结果图，文件名已按顺序编号。

| 文件 | 内容 | 对应提交要求 |
|---|---|---|
| `01_环境与labelImg启动.png` | 安装/启动 labelImg 的终端 + labelImg 界面 | Level 1 环境配置 |
| `02_labelImg标注界面.png` | labelImg 打开数据集，给可乐瓶画框并选择 cola | Level 2 标注界面 |
| `03_标注进度检查.png` | 标注进度统计：train 751/759、val 0/190、各类框数 | Level 2 标注数量 |
| `04_data_yaml配置.png` | VS Code 中 `data.yaml` 的完整内容 | Level 3 数据集配置 |
| `05_训练启动_数据与GPU.png` | 训练启动：601/150 张、GPU RTX 5060、Ultralytics 参数 | Level 3 训练开始 |
| `06_训练开始_epoch进度.png` | 50 epochs 开始，前几个 epoch 的 loss/mAP 进度 | Level 3 训练开始 |
| `07_训练完成_指标与权重.png` | 训练完成、best.pt 复验的 per-class 指标、权重路径 | Level 3 结果/权重 |
| `08_训练曲线_results.png` | train/val loss、precision、recall、mAP 曲线 | Level 3 结果图 |
| `09_混淆矩阵.png` | 归一化混淆矩阵 | Level 3 结果图 |
| `10_新图片推理终端.png` | 对新图片执行 predict，终端输出类别/置信度/框 | Level 4 |
| `11_新图片检测结果.png` | 新图片检测结果（带框和置信度） | Level 4 |
| `12_数据分布_labels.png` | 数据集类别分布、框尺寸分布 | Level 1/2 数据检查 |
| `13_PR曲线.png` | Precision-Recall 曲线 | Level 3 结果图 |
| `14_F1曲线.png` | F1-Confidence 曲线 | Level 3 结果图 |
| `15_标注样例_12张.jpg` | 12 张已标注图片的框+类别可视化 | Level 2 标注效果 |
| `16_预测样例_8张.jpg` | 8 张新图片的预测结果拼图 | Level 4 |
| `17_新图片检测_高清.jpg` | 新图片检测结果原图（1224×576） | Level 4 |

> `05`/`06`/`07` 来自训练时的终端截图；`08`/`09`/`13`/`14` 是训练结果图的屏幕截图，
> 原始文件同时在 `runs/train_20261018_1701/` 下。
