# 基于 YOLO 的机器人场景目标检测

| 提交信息 | 内容 |
|---|---|
| 姓名 | 王家铭 |
| 专业与班级 | 冶卓251 |
| 学号 | 202510201632 |

## 项目范围

对应仓库 YOLO 方向，目标为 Level 1 至 Level 4。使用本人在 LabelImg 中手工标注的 63 张图片，51 张训练、12 张验证，YOLO11n 在 CPU 上训练 30 轮，最后对另外 9 张图片检测。未制作 Level 5 页面。

详细过程、指标、限制和截图见 [report.md](report.md)。待补项目见 [提交检查清单.md](提交检查清单.md)。

## 快速复现

推荐与原实验一致的 Python 3.8.10。命令在本文件夹内执行：

```bash
python -m pip install -r requirements.txt
python train.py --check
python predict.py --check
python predict.py
```

`predict.py` 默认使用已提交的 `runs/manual_train/weights/best.pt`，处理 `new_images/`，输出到 `results/reproduce_predict/`。带框图片、检测标签及置信度会保存。已有实验结果在 `results/new_images/`，复现默认不覆盖它们。

重新训练：

```bash
python train.py --epochs 30
```

首次需联网下载官方 `yolo11n.pt` 初始化权重；也可以用 `--model 本地权重路径`。程序运行时将 data.yaml 的相对路径转换为本机绝对路径，文件夹重命名后仍可运行。下载或安装若遇代理问题，请参考报告中本机的问题记录，不要关闭证书验证。

## 目录

```text
学号/
├── README.md
├── report.md
├── train.py
├── predict.py
├── common.py
├── data.yaml
├── requirements.txt
├── dataset/
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── new_images/                  # 9 张训练外输入
├── runs/manual_train/           # 原始训练产物和 best.pt / last.pt
├── results/
│   ├── new_images/              # 真实检测结果
│   └── screenshots/             # 本人实际操作截图
├── provenance/                  # 数据来源、划分、原实验入口
└── 提交检查清单.md
```

包含实际使用的 63 张数据副本，不要求审阅者另外下载 949 张原始图片；没有复制其他同学的项目代码、模型或标签。
