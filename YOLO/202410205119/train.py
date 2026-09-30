#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py —— 使用 Ultralytics YOLOv8-n 训练 obstacle / cola / football 三分类检测模型
================================================================================
位置: 你的学号/train.py

【本脚本做什么】
把 data.yaml 指定的自制数据集（train 4 张 / val 2 张）喂给 YOLOv8-n 做迁移学习，
训练出能识别三类目标的权重文件 best.pt，并把训练过程与结果全部输出到 runs/ 目录。

【Windows 一键执行命令】（在本文件所在目录下的 PowerShell / CMD 里执行）
    :: 1) 先做训练前自检（不训练，几秒钟，用来排错）
    python train.py --check-only

    :: 2) 正式训练（CPU 推荐配置，本机已实测跑通）
    python train.py --epochs 100 --imgsz 640 --batch 4 --device cpu

    :: 3) 快速冒烟测试（只想确认流程通不通，用时最短）
    python train.py --epochs 5 --imgsz 320 --batch 2 --device cpu

    :: 4) 有 NVIDIA 显卡时改用 GPU（速度提升一个数量级以上）
    python train.py --epochs 200 --imgsz 640 --batch 16 --device 0

【训练产物】（由 Ultralytics 自动创建，全部位于 runs/detect/<实验名>/ 下）
    weights/best.pt           验证集上表现最好的权重（predict.py 用的就是它）
    weights/last.pt           最后一轮权重（用于 --resume 断点续训）
    args.yaml                 本次训练用到的全部超参（复现实验靠它）
    results.csv               每一轮的 box/cls/dfl loss 与 P / R / mAP 数值
    results.png               上述指标随轮数变化的曲线图
    confusion_matrix.png      混淆矩阵
    BoxPR_curve.png 等        PR / P / R / F1 曲线
    labels.jpg                数据集标签分布统计
    train_batch0.jpg          训练批次可视化（含增强后的框）
    val_batch0_labels.jpg     验证集真实标签可视化
    val_batch0_pred.jpg       验证集预测结果可视化

【路径规范】只使用相对路径 + 基于 __file__ 的动态路径，无任何绝对路径。
"""

# 让类型注解在旧版本 Python 上也能正常解析（不产生运行时代价）
from __future__ import annotations

import argparse          # 解析命令行参数
import os                # 路径与系统相关操作
import sys               # 退出码控制
from pathlib import Path # 面向对象的路径操作，比字符串拼接更安全


# ---------------------------------------------------------------------------
# 0. 路径常量：全部基于本文件位置动态推导，保证可移植
# ---------------------------------------------------------------------------
# __file__ 是本脚本自身的路径；.resolve() 转成绝对路径并解析符号链接；
# .parent 取所在目录，也就是项目根目录（你的学号/）。
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data.yaml"          # 数据集配置文件的绝对路径

# 允许被识别为图片的后缀（收集图片时用）
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。

    把所有常用超参都暴露成命令行选项，好处是：
      - 调参时不用改代码，直接换命令即可；
      - 每次实验用了什么参数一目了然，便于写进报告。
    """
    parser = argparse.ArgumentParser(
        description="在自制三分类数据集上训练 YOLOv8-n 目标检测模型",
        # 让 --help 自动显示每个参数的默认值
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ===================== 模型与数据 =====================
    # --model: 预训练权重文件。
    #   yolov8n.pt = YOLOv8 nano，官方在 COCO（80 类）上训练好的最小模型
    #   （约 3.2M 参数 / 6MB）。首次运行会自动下载并缓存到本地。
    #   用预训练权重做迁移学习，是本项目只有 6 张图也能出结果的关键：
    #   backbone/neck 直接继承 COCO 学到的边缘、纹理、轮廓等通用视觉特征。
    #   若想从随机初始化开始训（本项目不推荐，6 张图必然学不到东西），可传 yolov8n.yaml。
    parser.add_argument("--model", default="yolov8n.pt",
                        help="预训练权重；yolov8n.pt 即 YOLOv8-n")

    # --data: 数据集配置文件路径，默认指向同目录的 data.yaml
    parser.add_argument("--data", default=str(DATA_YAML),
                        help="数据集配置文件路径")

    # ===================== 训练规模 =====================
    # --epochs: 训练轮数。整个训练集被完整学习一遍算 1 个 epoch。
    #   样本量很小（4 张），轮数太少模型来不及收敛，故默认 100。
    parser.add_argument("--epochs", type=int, default=100, help="训练总轮数")

    # --batch: 批大小（一次前向/反向同时处理几张图）。
    #   只在小数据集里设为 2~4，保证每个 batch 内能看到多个目标框。
    parser.add_argument("--batch", type=int, default=4,
                        help="批大小；-1 表示由框架自动估计")

    # ===================== 输入尺寸 =====================
    # --imgsz: 训练前把图片统一缩放成 imgsz × imgsz 的正方形再送进网络。
    #   640 是 YOLOv8 官方默认值，也是精度与速度的平衡点。
    #   瓶颈在内存/显存时可以降到 416 或 320；纯 CPU 冒烟测试建议 320。
    parser.add_argument("--imgsz", type=int, default=640,
                        help="网络输入边长（会被缩放成正方形）")

    # ===================== 设备与数据加载 =====================
    # --workers: DataLoader 读图子进程数。
    #   Windows 上子进程会用 spawn 方式重新导入主模块，在复杂环境里容易卡死
    #   或报 freeze_support 错误，因此 Windows 默认给 0（主进程读图，慢但稳定）；
    #   Linux / Colab 可以给 8 来加速。
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 8,
                        help="数据加载子进程数（Windows 建议 0）")

    # --device: 训练设备。"" = 自动选择（有 GPU 用 GPU）；
    #   "cpu" = 强制 CPU；"0" = 使用第 0 号 GPU；"0,1" = 用两张卡。
    parser.add_argument("--device", default="",
                        help="cpu / 0 / 0,1；留空为自动")

    # ===================== 优化器超参 =====================
    # --lr0: 初始学习率。0.01 是 YOLOv8 配 SGD 的官方默认值；
    #   小数据集微调时可调小（如 0.001），避免一步就把预训练权重"冲坏"。
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")

    # --lrf: 最终学习率系数，lr_final = lr0 × lrf。
    #   配合余弦退火，让训练后期的更新步长逐渐变小，收敛更稳。
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="最终学习率系数（lr_final = lr0 * lrf）")

    # --momentum: SGD 动量，利用历史梯度平滑更新方向、抑制震荡。
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD 动量")

    # --weight-decay: 权重衰减（L2 正则），抑制权重过大，缓解过拟合。
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="权重衰减系数")

    # --optimizer: 优化器类型。auto 让框架根据迭代次数自动选择。
    parser.add_argument("--optimizer", default="auto",
                        help="优化器: auto / SGD / Adam / AdamW")

    # --patience: 早停耐心值。连续 N 轮验证指标不再提升就提前结束训练，
    #   避免无意义地耗时间。
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")

    # ===================== 数据增强 =====================
    # 只有 6 张图，必须靠数据增强"人造"出更多变化，否则必然严重过拟合。
    #
    # --fliplr: 左右翻转概率。可乐瓶、足球左右翻转后语义不变，是安全增强。
    #   （若目标是文字/数字，翻转会改变语义，就不能开。）
    parser.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率")

    # --flipud: 上下翻转概率。本项目目标摆放正常朝上，上下翻转意义不大，故 0。
    parser.add_argument("--flipud", type=float, default=0.0, help="上下翻转概率")

    # --mosaic: 把 4 张图随机拼成 1 张，让模型见到更多背景组合与目标尺度。
    parser.add_argument("--mosaic", type=float, default=1.0, help="mosaic 四图拼接概率")

    # --hsv-h/s/v: 随机抖动色调/饱和度/明度，模拟不同光照条件。
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV 色调抖动幅度")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV 饱和度抖动幅度")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV 明度抖动幅度")

    # --degrees: 随机旋转角度范围。目标摆放较正，故 0。
    parser.add_argument("--degrees", type=float, default=0.0, help="随机旋转角度范围")

    # --scale / --translate: 随机缩放增益与随机平移比例，
    #   提升模型对目标大小和位置变化的鲁棒性。
    parser.add_argument("--scale", type=float, default=0.5, help="随机缩放增益范围")
    parser.add_argument("--translate", type=float, default=0.1, help="随机平移比例")

    # --close-mosaic: 最后 N 轮关闭 mosaic。
    #   mosaic 拼出来的图与真实图分布有差异，训练末尾关掉它让模型在真实
    #   分布上做最后收敛，通常能提升最终精度。
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="最后 N 轮关闭 mosaic")

    # ===================== 输出与可复现性 =====================
    # --project / --name: 输出目录 = project/name，即 runs/detect/train。
    #   重复运行时框架会自动改成 train2、train3…… 除非加 --exist-ok。
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "detect"),
                        help="训练输出根目录")
    parser.add_argument("--name", default="train", help="本次实验的子目录名")
    parser.add_argument("--exist-ok", action="store_true",
                        help="允许覆盖同名输出目录")

    # --seed: 随机种子。固定后数据顺序、增强随机数、权重初始化都可复现，
    #   报告里给出的指标才有对比意义。
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")

    # --deterministic: 启用确定性算法，配合 seed 使用。
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="启用确定性算法")

    # --resume: 从 last.pt 继续训练（训练中途被打断时用）。
    parser.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")

    # ===================== 验证与绘图 =====================
    # --val: 每个 epoch 结束后都在验证集上评估一次，并据此选出 best.pt。
    #   验证集不参与梯度更新，所以它的指标才能反映泛化能力。
    parser.add_argument("--val", action="store_true", default=True,
                        help="每轮在验证集上评估")

    # --plots: 生成 results.png / 混淆矩阵 / PR 曲线等图。
    parser.add_argument("--plots", action="store_true", default=True,
                        help="生成训练曲线图")

    # ===================== 仅自检 =====================
    # --check-only: 只检查环境和数据集，不真正开始训练。
    #   训练很耗时，用这个开关可以在几秒内把低级错误全部挡掉。
    parser.add_argument("--check-only", action="store_true",
                        help="只检查环境与数据集，不训练（用于快速排错）")

    return parser.parse_args()


def check_env_and_dataset(args: argparse.Namespace) -> int:
    """训练前自检：把最常见的失败原因提前暴露，而不是等 YOLO 抛出一长串栈。

    检查 5 项：
      1) ultralytics 能否导入、版本号；
      2) torch 是否可用、CUDA 是否可用（决定训练速度预期）；
      3) data.yaml 是否存在、能否按 UTF-8 解析、nc 与 names 是否一致；
      4) images/train 与 images/val 目录是否存在、各有几张图；
      5) 每张图片是否都有同名 .txt 标签（图片与标签必须一一对应）。

    返回值：成功配对的「图片/标签」数量。为 0 时不应继续训练。
    """
    print("=" * 74)
    print(" 训练前自检 (pre-flight check)")
    print("=" * 74)

    # ---------- 1) ultralytics ----------
    try:
        import ultralytics                       # noqa: F401  仅为检查能否导入
    except ImportError:
        print("[X] 未安装 ultralytics。请先执行: pip install -r requirements.txt")
        return 0
    print(f"[OK] ultralytics 版本 : {ultralytics.__version__}")

    # ---------- 2) torch / CUDA ----------
    try:
        import torch
        cuda = torch.cuda.is_available()         # True 表示检测到可用 NVIDIA GPU
        print(f"[OK] torch 版本       : {torch.__version__}")
        print(f"[OK] CUDA 可用        : {cuda}")
        if not cuda:
            # 这不是错误，只是速度预期提示：CPU 上完整训练会慢很多
            print("     [!] 未检测到可用 GPU，将回退到 CPU 训练。")
            print("         建议先用小规模跑通流程: --epochs 5 --imgsz 320 --batch 2")
            print("         或把项目放到 Colab / Linux 的 GPU 环境训练。")
    except ImportError:
        print("[!] 未安装 torch，无法检查 CUDA（ultralytics 会自动安装它）")

    # ---------- 3) data.yaml ----------
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"[X] 找不到数据集配置文件: {data_path}")
        return 0
    print(f"[OK] data.yaml        : {data_path}")

    try:
        import yaml
        # 显式用 UTF-8 打开：Windows 默认用 ANSI(GBK) 解码含中文的 yaml 会直接抛异常
        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[X] data.yaml 解析失败: {e!r}")
        print("    常见原因：文件被存成 GBK 编码，或缩进里混用了 Tab。请另存为 UTF-8。")
        return 0

    names = cfg.get("names", {})
    nc = cfg.get("nc", None)
    # names 允许写成列表形式，这里统一转成 {id: name} 字典便于比较
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    print(f"[OK] nc               : {nc}")
    print(f"[OK] names            : {names}")

    # nc 与 names 条目数必须一致 —— 这是最常见也最难定位的配置错误之一
    if nc is not None and len(names) != nc:
        print(f"[X] nc={nc} 但 names 有 {len(names)} 项，二者必须相等！")
        return 0
    # 三个类别编号必须齐全
    if not {0, 1, 2}.issubset(set(names.keys())):
        print("[X] names 缺少 0/1/2 中的某个类别编号，请检查 data.yaml。")
        return 0

    # ---------- 4) + 5) 图片目录与标签配对 ----------
    # 目录按 data.yaml 所在目录解析，与 Ultralytics 的实际行为保持一致
    data_root = data_path.resolve().parent
    total_pairs = 0                          # 累计成功配对的图片数

    for split in ("train", "val"):
        rel = cfg.get(split)
        if rel is None:
            print(f"[X] data.yaml 中缺少 '{split}:' 配置项")
            return 0

        img_dir = (data_root / rel).resolve()
        # YOLO 约定：把路径中的 /images/ 换成 /labels/ 就是标签目录
        lbl_dir = Path(str(img_dir).replace(
            os.sep + "images" + os.sep, os.sep + "labels" + os.sep))

        if not img_dir.is_dir():
            print(f"[X] {split} 图片目录不存在: {img_dir}")
            return 0

        # 只统计允许的图片后缀，避免把 .cache 等文件算进去
        images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS)
        print(f"[OK] {split:5s} 图片目录 : {img_dir}  （{len(images)} 张）")

        if not images:
            print(f"     [!] {split} 目录为空，请先放入图片。")

        # 逐张检查是否有同名标签文件
        missing = []                          # 记录缺标签的图片名
        for img in images:
            if (lbl_dir / f"{img.stem}.txt").is_file():
                total_pairs += 1              # 配对成功
            else:
                missing.append(img.stem + ".txt")
        if missing:
            # 没有标签的图片会被 YOLO 当作"背景图"（负样本）使用，
            # 少量可以接受，数量多说明标注或命名有问题，必须警告。
            print(f"     [!] {len(missing)} 张图缺少同名标签: {missing[:5]}"
                  f"{' ...' if len(missing) > 5 else ''}")

    print(f"[OK] 有效的 图片/标签 配对数量: {total_pairs}")
    print("=" * 74)
    return total_pairs


def main() -> None:
    """主流程：自检 → 加载模型 → 训练 → 打印结果位置。"""
    args = parse_args()                      # 1) 解析命令行参数

    # 把当前工作目录切到项目根目录。
    # 目的：让 runs/ 的输出位置、预训练权重的下载位置都稳定在项目内，
    # 不受"你从哪个目录敲的 python train.py"影响。
    os.chdir(PROJECT_ROOT)

    pairs = check_env_and_dataset(args)      # 2) 训练前自检

    # 只自检模式：检查完就退出，不训练
    if args.check_only:
        print("--check-only 已指定：自检完成，退出（不进行训练）。")
        return

    # 一张有效样本都没有就不要再往下走了，训练必然失败
    if pairs == 0:
        print("[X] 没有任何有效的 图片/标签 配对，训练必然失败。")
        print("    请按 data.yaml 的 train/val 路径放入图片与同名标签。")
        sys.exit(1)

    # 自检通过后才导入 YOLO：这样缺依赖时给的是中文提示，而不是一长串报错栈
    from ultralytics import YOLO

    # =======================================================================
    # 3. 加载模型（迁移学习：加载 COCO 预训练权重）
    # =======================================================================
    # YOLO("yolov8n.pt") 会读取官方预训练权重：
    #   - backbone / neck 直接继承 COCO 上学到的通用特征提取能力；
    #   - head 的输出通道在首次读取 data.yaml 时自动由 80 类改成 3 类（nc=3）。
    print(f"[1/3] 加载模型: {args.model}")
    model = YOLO(args.model)

    # =======================================================================
    # 4. 开始训练
    # =======================================================================
    # 下面传入的所有参数，Ultralytics 都会原样记录到
    # runs/detect/<name>/args.yaml，方便日后精确复现这次实验。
    print(f"[2/3] 开始训练: epochs={args.epochs}, imgsz={args.imgsz}, "
          f"batch={args.batch}, device={args.device or 'auto'}")
    results = model.train(
        # ---- 数据与训练规模 ----
        data=args.data,            # 数据集配置（图片/标签位置、nc、names）
        epochs=args.epochs,        # 训练轮数
        batch=args.batch,          # 批大小
        imgsz=args.imgsz,          # 输入分辨率（正方形边长）
        # ---- 设备与数据加载 ----
        device=args.device or None,   # None = 自动选择（有 GPU 用 GPU，否则 CPU）
        workers=args.workers,         # 读图子进程数
        # ---- 优化器 ----
        optimizer=args.optimizer,
        lr0=args.lr0,                 # 初始学习率
        lrf=args.lrf,                 # 最终学习率系数
        momentum=args.momentum,       # SGD 动量
        weight_decay=args.weight_decay,
        # warmup：前 3 轮用很小的学习率"热身"，避免初期大梯度破坏预训练权重
        warmup_epochs=3.0,
        patience=args.patience,       # 早停耐心值
        # ---- 数据增强（小样本必备）----
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        # ---- 输出与可复现性 ----
        project=args.project,         # 输出根目录（runs/detect）
        name=args.name,               # 本次实验子目录名（train）
        exist_ok=args.exist_ok,       # 是否允许覆盖同名目录
        seed=args.seed,               # 随机种子
        deterministic=args.deterministic,
        resume=args.resume,           # 是否断点续训
        # ---- 验证与可视化 ----
        val=args.val,                 # 每轮在验证集上评估
        plots=args.plots,             # 生成 results.png / 混淆矩阵等
        # ---- 缓存 ----
        # 数据集很小，cache=True 把图片一次性读进内存，省掉每轮的磁盘 IO
        cache=True,
    )

    # =======================================================================
    # 5. 打印结果位置，并提示下一步
    # =======================================================================
    # results.save_dir 是框架实际写入的目录（一般是 runs/detect/train）
    save_dir = Path(getattr(results, "save_dir",
                            PROJECT_ROOT / "runs" / "detect" / args.name))
    best = save_dir / "weights" / "best.pt"      # 最佳权重路径
    print(f"\n[3/3] 训练结束，输出目录: {save_dir}")
    print(f"      最佳权重: {best}  （存在: {best.is_file()}）")
    print("      下一步：")
    print("        1) 查看 runs 目录结果:  dir runs\\detect\\train")
    print("        2) 用最佳权重做推理:    python predict.py --source <图片或目录>")


# 只有当本文件被直接运行时才执行 main()。
# 作用：Windows 上若把 workers 调到 >0，DataLoader 的子进程会重新导入本模块，
#       没有这个守卫就会无限递归创建进程。
if __name__ == "__main__":
    main()
