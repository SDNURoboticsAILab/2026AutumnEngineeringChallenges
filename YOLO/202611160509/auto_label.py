# -*- coding: utf-8 -*-
"""
自动标注脚本
============
功能：用训练好的模型对未标注图片自动生成 YOLO 标签，并整理为
      与 dataset 相同的标准数据集结构（images/train|val、labels/train|val）。

用法：
    python auto_label.py --source 未标注图片文件夹
    python auto_label.py --source 未标注图片文件夹 --val-ratio 0.2 --conf 0.25

输出：dataset_auto（images + labels + data.yaml），可直接用于训练
"""
import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

# ---------- 路径配置 ----------
BASE = Path(__file__).resolve().parent                          # 项目根目录
DEFAULT_WEIGHTS = BASE / "runs" / "weights" / "best.pt"
RUNS_DIR = BASE / "runs"                                        # 预测输出目录

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["cola", "football", "obstacle"]                      # 类别（与 data.yaml 一致）


def main():
    parser = argparse.ArgumentParser(description="YOLO 自动标注")
    parser.add_argument("--source", required=True, help="待标注图片文件夹（必填）")
    parser.add_argument("--out", default=str(BASE / "dataset_auto"), help="输出数据集目录")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（保证划分可复现）")
    args = parser.parse_args()

    src = Path(args.source)
    assert src.is_dir(), f"输入文件夹不存在: {src}"
    assert Path(args.weights).exists(), f"权重不存在: {args.weights}"
    out = Path(args.out)
    labels_dir = out / "labels"

    # 1) 批量预测（输出带框预览图 + YOLO 标签）
    model = YOLO(args.weights)
    model.predict(
        source=str(src),
        imgsz=640,
        conf=args.conf,
        device=0,
        save=True,
        save_txt=True,
        save_conf=False,       # 数据集标签不包含置信度
        line_width=3,
        workers=0,
        project=str(RUNS_DIR),
        name="auto_label",
        exist_ok=True,
    )
    pred_labels = RUNS_DIR / "auto_label" / "labels"

    # 2) 收集标签到输出目录（无检测的图片生成空标签）
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    for img in images:
        txt = pred_labels / (img.stem + ".txt")
        dst = labels_dir / (img.stem + ".txt")
        if txt.exists():
            shutil.copy2(txt, dst)
        else:
            dst.touch()

    # 3) 划分 train/val 并整理成标准结构
    rng = random.Random(args.seed)
    rng.shuffle(images)
    n_val = max(1, round(len(images) * args.val_ratio))
    val_set = set(images[:n_val])
    for split, imgs in (("train", images[n_val:]), ("val", val_set)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.move(str(img), str(out / "images" / split / img.name))
            shutil.move(str(labels_dir / (img.stem + ".txt")),
                        str(labels_dir / split / (img.stem + ".txt")))

    # 4) classes.txt（与 dataset 一致，labels/train 与 labels/val 各一份）
    for sub in ("train", "val"):
        (labels_dir / sub / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    # 5) 生成 data.yaml
    (out / "data.yaml").write_text(
        "# YOLO 自动标注数据集配置\n"
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        + "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES)) + "\n",
        encoding="utf-8",
    )

    print(f"完成：共 {len(images)} 张，train={len(images) - n_val}，val={n_val}")
    print(f"数据集已整理到: {out}")


if __name__ == "__main__":
    main()
