# dataset

本目录是标准 YOLO 检测数据集，共 **751 张**图片，全部有同名 `.txt` 标签。

## 数量

| split | images | labels | boxes | obstacle | cola | football |
|---|---|---|---|---|---|---|
| train | 601 | 601 | 662 | 262 | 189 | 211 |
| val | 150 | 150 | 162 | 61 | 47 | 54 |
| **合计** | **751** | **751** | **824** | **323** | **236** | **265** |

- 划分方式：按类别分层随机 8:2，随机种子 42；
- 逐张清单见 `dataset_split.csv`（列：new_name, split, class, class_id, source_image, boxes）。

## 结构

```text
dataset/
├─ images/
│  ├─ train/   601 张 .jpg
│  └─ val/     150 张 .jpg
├─ labels/
│  ├─ train/   601 个 .txt
│  └─ val/     150 个 .txt
├─ classes.txt
└─ dataset_split.csv
```

图片与标签同名成对：`images/train/cola_0001.jpg` ↔ `labels/train/cola_0001.txt`。

## 标签格式

每行一个目标：

```text
类别id  中心x  中心y  宽  高
```

坐标是相对图片宽高的归一化值（0~1）。

类别：

```text
0 obstacle   蓝色方块障碍物
1 cola       可乐瓶
2 football   足球
```

## 数据来源

原始数据是仓库 `YOLO/obstacle|cola|football` 下的 949 张图片：

- obstacle 316、cola 293、football 340；
- 其中初次下载缺的 28 张 cola 已按 manifest 补齐；
- 人工标注完成 751 张（train 751/759，val 0/190）；
- 为了让训练/验证都有标签，取已标注的 751 张重新按类别分层切 8:2，得到 601/150；
- 剩余 198 张（原 val 190 + 原 train 8 张漏标/空场景）未放入本数据集。

## 说明

`labels/train/classes.txt` 和 `labels/val/classes.txt` 是标注工具 labelImg 读取标签时用的类别文件，
训练时不会被当作图片标签。训练扫描结果为 `0 corrupt, 0 background`。
