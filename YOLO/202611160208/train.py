# -*- coding: utf-8 -*-
"""YOLO 模型训练脚本（Level 3）

用法:
  python train.py                    # 默认 yolov8s, 100 epochs
  python train.py --name exp1 --epochs 150 --batch 16

训练结果输出到 runs/<实验名>/
"""
import argparse
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt",
                    help="预训练权重：yolov8n/s/m 或 yolo11n/s/m（n小快、m大准）")
    ap.add_argument("--epochs", type=int, default=100, help="训练轮数")
    ap.add_argument("--imgsz", type=int, default=640, help="输入分辨率（与图片宽度一致）")
    ap.add_argument("--batch", type=int, default=16, help="批大小，12GB显存可到16~32")
    ap.add_argument("--name", default="baseline", help="本次实验名（结果目录名）")
    ap.add_argument("--patience", type=int, default=30, help="早停耐心值（0=禁用早停）")
    args = ap.parse_args()

    model = YOLO(args.model)  # 首次运行自动下载预训练权重
    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=os.path.join(ROOT, "runs"),
        patience=args.patience,  # 无提升 N 轮则早停，防过拟合
        device=0,           # 用 GPU 0
        workers=4,
    )

    # 训练结束后在验证集上评估一次，打印 mAP 等指标
    metrics = model.val()
    print(f"\n=== 验证集指标 ===")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
