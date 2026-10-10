# train.py
# 命令行训练入口（GUI 内的训练逻辑在 tab_train.py）
import argparse
from pathlib import Path

from ultralytics import YOLO

from core import PROJECT_ROOT


DEFAULT_DATA = str(PROJECT_ROOT / "data.yaml")
DEFAULT_WEIGHTS = str(PROJECT_ROOT / "yolo11n.pt")
DEFAULT_PROJECT = str(PROJECT_ROOT / "runs" / "train")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 训练脚本")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="exp")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml 不存在: {args.data}")
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"权重不存在: {args.weights}")

    print("=" * 60)
    print("YOLO 训练")
    print("=" * 60)
    print(f"  data    = {args.data}")
    print(f"  weights = {args.weights}")
    print(f"  epochs  = {args.epochs}")
    print(f"  imgsz   = {args.imgsz}")
    print(f"  batch   = {args.batch}")
    print(f"  device  = {args.device}")
    print(f"  project = {args.project}/{args.name}")
    print("=" * 60)

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        lr0=0.01, lrf=0.01, cls=1.5,
        degrees=10.0, translate=0.15, scale=0.5,
        fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=0.5, mixup=0.05, patience=15,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n训练完成，权重: {best}")


if __name__ == "__main__":
    main()