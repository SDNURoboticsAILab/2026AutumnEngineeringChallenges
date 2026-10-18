# results

## predict/

从 `dataset/images/val` 之外的新图片中挑选的 12 张推理结果（带框、类别、置信度）：

- cola：`cola_0003.jpg`、`cola_0036.jpg`、`cola_0059.jpg`、`cola_0004.jpg`
- football：`football_0001.jpg`、`football_0007.jpg`、`football_0011.jpg`、`football_0012.jpg`
- obstacle：`obstacle_0006.jpg`、`obstacle_0011.jpg`、`obstacle_0012.jpg`、`obstacle_0020.jpg`

完整 190 张推理结果在本地工作目录 `results/predict/`，仓库中只放了 12 张代表样例。

另外还有 3 张后来补充的独立新图结果：

- `new_test_01_cola_bottle.jpg`：近景可乐瓶
- `new_test_02_cubes_balls.jpg`：多个蓝方块 + 足球
- `new_test_03_two_balls.jpg`：两个足球 + 蓝方块

详细输出和误检说明见 `predict/README_新增.md`。

## screenshots/

各阶段截图/结果图，编号 01~17，说明见 `screenshots/README.md`。

## new_test/

一张完全独立的新图片（1224×576）：左边蓝色方块、中间红标可口可乐瓶、右边足球。

| 目标 | 结果 | 置信度 |
|---|---|---|
| football | ✅ | 0.78 |
| obstacle | ✅ | 0.45 |
| cola | ❌ 漏检 | — |

3 个目标检出 2 个。cola 漏检的详细分析见 `new_test/README.md`：
训练集里的 cola 主要是深色瓶身，这张是红标透明瓶，属于训练分布外；
单独裁剪时模型只对瓶身下半段深色液体有 0.56 响应，整图漏检。

文件：

- `new_test/scene_input.jpg`：输入原图
- `new_test/result_640.jpg`：检测结果图
- `new_test/prediction_result.txt`：终端输出
- `new_test/README.md`：完整分析
