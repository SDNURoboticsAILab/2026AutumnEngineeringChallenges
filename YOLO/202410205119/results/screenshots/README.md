# 截图清单（照着截，文件名必须一致）

> 把截好的图**全部放进本目录**（`results/screenshots/`）。
> 文件名按下表写死，`report.md` 里已经预留对应的引用位置，放进去就直接生效。

## Windows 截图方法（3 种，任选）

| 方法 | 操作 | 说明 |
|---|---|---|
| **① 快捷键（推荐）** | 按 `Win + Shift + S` → 鼠标框选区域 → 自动复制到剪贴板 | 最方便，截完要粘贴到画图里另存 |
| **② 截图工具** | 开始菜单搜索「截图工具」→ 点「新建」→ 框选 | 可直接保存为文件 |
| **③ PrintScreen** | 按 `PrtScn` 全屏复制 → 到画图里 `Ctrl+V` → 保存 | 截全屏 |

**保存为标准做法**：`Win + Shift + S` 框选后 → 打开「画图」(开始菜单搜 `mspaint`) →
`Ctrl + V` 粘贴 → `Ctrl + S` 保存为 PNG → 选到本目录、填上下表文件名。

---

## 当前状态（已归档 8 张，仅剩 1 张待补）

| # | 文件名 | 对应 Level | 状态 |
|---|---|---|---|
| 1 | `L1_env.png` | Level 1 环境安装成功 | ✅ 已提供 |
| 2 | `L2_labeled.png` | Level 2 标注好的图片 | ✅ 已提供 |
| 3 | `L2_labeltxt.png` | Level 2 标签文件内容 | ✅ 已提供 |
| 4 | `L3_train_start.png` | Level 3 训练进行中 | ✅ 已提供 |
| 5 | `L3_train_epochs.png` | Level 3 训练后段 | ✅ 已提供 |
| 6 | `L3_train_done.png` | Level 3 训练结束 | ✅ 已提供 |
| 7 | `L3_weights.png` | Level 3 权重文件信息 | ✅ 已提供 |
| 8 | `L4_predict.png` | Level 4 推理输出 | ✅ 已提供 |
| 9 | `L5_web.png` | Level 5 网页界面 | ⚠️ **待补** |

> 这 8 张已全部嵌入 `../../report.md` 的**附录 D：各 Level 验证截图**。
> 只剩 Level 5 的网页界面截图需要在本地启动服务后补截。

---

## 需要补截的 1 张：`L5_web.png`

在项目根目录打开终端，执行：

```bat
python results\code\gradio_app.py
```

然后：

1. 终端会打印 `Running on local URL: http://127.0.0.1:7860`，浏览器打开该地址；
2. 点左侧「上传图片」框 → 选 `dataset\images\val\img06.jpg`；
3. **把「置信度阈值」滑块往左拖到 0.10~0.20**（关键，默认 0.25 检不到东西）；
4. 等右侧出现检测结果图后截图。

**截图画面要同时包含四样**（对应 Level 5 验收要求）：

| 需要出现在截图里 | 对应验收要求 |
|---|---|
| 左侧上传的原图 | 页面能够正常上传图片 |
| 右侧检测结果图（带框和标签） | 正确显示检测后的图片 |
| 「识别结果明细」表格 | 显示目标类别与置信度 |
| 「检测摘要」文本框 | 汇总检测结果 |

建议用 `Alt + PrtScn` 截整个窗口，或框选整个浏览器页面。

截好后存为本目录下的 `L5_web.png`。

---

## 以下是各张图的原始操作说明（备查）

## 需要截的 8 张图

| # | 文件名 | 对应 Level | 截什么 |
|---|---|---|---|
| 1 | `L1_env.png` | Level 1 | 终端里环境安装成功的版本输出 |
| 2 | `L2_labeled.png` | Level 2 | 画着目标框的标注图 |
| 3 | `L2_labeltxt.png` | Level 2 | 标签 `.txt` 文件内容 |
| 4 | `L3_train_start.png` | Level 3 | 训练过程的终端输出 |
| 5 | `L3_train_epochs.png` | Level 3 | 训练后段（指标趋稳） |
| 6 | `L3_train_done.png` | Level 3 | 训练结束提示 |
| 7 | `L3_weights.png` | Level 3 | 权重文件信息 |
| 8 | `L4_predict.png` | Level 4 | 推理终端输出（类别+置信度） |

---

## 逐张操作说明

### 1. `L1_env.png` —— 环境安装成功

在项目目录打开终端，执行：

```bat
cd /d "C:\Users\zhonglilsq\Desktop\论文\2026XXXXXX"
python -c "import ultralytics, torch, cv2; print('ultralytics', ultralytics.__version__); print('torch', torch.__version__); print('opencv', cv2.__version__); print('CUDA', torch.cuda.is_available())"
```

把输出结果截图。**画面里要能看到 ultralytics / torch / opencv 的版本号。**

### 2. `L1_tree.png` —— 项目目录结构

```bat
tree /F
```

目录树很长，截**前面一部分**即可（能看到 `dataset`、`runs`、`results` 和 6 个文件就行）。
或者用资源管理器打开项目目录、切到「查看 → 详细信息」截图。

### 3. `L2_labeled.png` —— 标注好的图片

执行以下命令生成带框的标注图（会在当前目录生成 `label_check/` 文件夹）：

```bat
python results\code\show_labels.py
```

然后打开 `label_check\train_img03.jpg`，截图。**画面里要能看到蓝色障碍方块上的红框。**

### 4. `L2_labeltxt.png` —— 标签文件内容

```bat
type dataset\labels\train\img03.txt
```

截图输出。**画面里要能看到 4 行、每行 5 个数字（类别号 + 4 个归一化坐标）。**

### 5. `L3_train_start.png` —— 训练启动

```bat
python train.py --epochs 60 --imgsz 320 --batch 2 --device cpu
```

趁训练开始、**还没跑完时**截图。画面里要能看到：
- `Overriding model.yaml nc=80 with nc=3`
- `Transferred 319/355 items from pretrained weights`
- 模型结构表的一部分

> 训练只要约 44 秒，所以建议**先开好截图工具、点「新建」准备好**，再回车运行。

### 6. `L3_train_done.png` —— 训练结束 + 权重信息

训练跑完后，同一个终端窗口继续：

```bat
dir runs\detect\train\weights
```

截图。**画面里要能看到 `best.pt` 和 `last.pt`，大小约 6,226,033 字节。**

### 7. `L4_predict.png` —— 推理结果

```bat
python predict.py --source dataset/images/val --conf 0.15 --print-dets
```

截图。**画面里要能看到类似这样的行：**

```text
img06.jpg | 类别=football(id=2) 置信度=0.626 框=(218,210)-(294,283)
```

### 8. `L5_web.png` —— 网页界面

```bat
pip install gradio
python results\code\gradio_app.py
```

浏览器打开 `http://127.0.0.1:7860`，
上传 `dataset\images\val\img06.jpg`，
**把「置信度阈值」滑块往左拖到 0.10~0.20**，
等右侧出现检测结果后截图。

**画面里要能看到四样东西**（对应 Level 5 验收要求）：
1. 左侧上传的原图
2. 右侧检测结果图（带框和标签）
3. 「识别结果明细」表格（类别 + 置信度）
4. 「检测摘要」文本框

---

## 截完后的检查

- [ ] 8 张图都放进 `results/screenshots/` 了
- [ ] 文件名和上表**完全一致**（大小写、下划线都不能错）
- [ ] 每张图都能看清文字（不要截太小、不要糊）
- [ ] 第 8 张里检测框清晰可见

---

## 关于置信度阈值的提醒

本项目只用了 6 张图片训练，模型置信度普遍偏低（最好的检测结果约 0.63）。
所以：

- 截图时 `--conf` 用 **0.15**，网页滑块拖到 **0.10~0.20**；
- 用 `--conf 0.25` 的话 `img02.jpg` 会完全检不到目标，`img06.jpg` 也只能检到 2 个；
- **这是数据量的客观限制，不是程序问题**，如实展示即可。
