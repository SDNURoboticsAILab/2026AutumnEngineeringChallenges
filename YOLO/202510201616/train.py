#!/usr/bin/env python
"""YOLO 目标检测训练脚本（提交版）。

用法：
    python train.py
    python train.py --epochs 100 --batch 16

数据集：dataset/images/{train,val} + dataset/labels/{train,val}
配置：data.yaml
输出：runs/train_YYYYmmdd_HHMM/，最佳权重 weights/best.pt
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

DATA_DIR = ROOT / "dataset"
DATA_YAML = ROOT / "data.yaml"


def count_labels(split):
    lbl_dir = DATA_DIR / "labels" / split
    files = [p for p in lbl_dir.glob("*.txt") if p.name != "classes.txt"]
    nonempty = [p for p in files if p.stat().st_size > 0]
    return len(files), len(nonempty)


def main():
    ap = argparse.ArgumentParser(description="训练 YOLO 目标检测模型")
    ap.add_argument("--model", default=str(ROOT / "weights" / "yolo11n.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto / 0 / cpu")
    ap.add_argument("--name", default="")
    ap.add_argument("--no-copy", action="store_true", help="不覆盖 weights/best.pt")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        print("找不到 data.yaml:", DATA_YAML)
        return 2

    train_total = len(list((DATA_DIR / "images" / "train").glob("*.jpg")))
    train_files, train_nonempty = count_labels("train")
    val_files, val_nonempty = count_labels("val")
    print(f"训练集: {train_total} 张图, {train_files} 个标签文件 ({train_nonempty} 个非空)")
    print(f"验证集: {val_files} 个标签文件 ({val_nonempty} 个非空)")

    if train_nonempty == 0:
        print("\n没有找到任何标注，请检查 dataset/labels/train。")
        return 2
    if train_files < train_total:
        print(f"\n注意：有 {train_total - train_files} 张训练图没有标签，会被当成背景图。")
    if val_nonempty == 0:
        print("\n注意：验证集没有标签，mAP 会一直是 0。")

    import torch
    from ultralytics import YOLO

    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == 0:
        print("使用 GPU:", torch.cuda.get_device_name(0))
    else:
        print("使用 CPU 训练（会比较慢）")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        print(f"本地没有权重 {model_path}，将让 Ultralytics 自动下载 yolo11n.pt")
        model = YOLO("yolo11n.pt")

    name = args.name or time.strftime("train_%Y%m%d_%H%M")
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=4,
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=False,
        pretrained=True,
        patience=20,
        seed=0,
        plots=True,
        val=True,
    )

    best = ROOT / "runs" / name / "weights" / "best.pt"
    print("\n训练结束。最佳权重:", best)
    if best.exists() and not args.no_copy:
        (ROOT / "weights").mkdir(exist_ok=True)
        import shutil

        shutil.copy2(best, ROOT / "weights" / "best.pt")
        print("已复制到:", ROOT / "weights" / "best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
