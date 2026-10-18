#!/usr/bin/env python
"""用自己训练的模型做目标检测（Level 4）。

用法：
    python predict.py --source 新图片.jpg
    python predict.py --source 新图片文件夹
    python predict.py --source 新图片.jpg --conf 0.3

结果图保存在 results/predict*/ 下，终端会打印每个目标的类别和置信度。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def latest_best():
    candidates = sorted(
        ROOT.glob("runs/**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser(description="YOLO 推理")
    ap.add_argument("--weights", default="", help="默认 weights/best.pt 或最新的 runs/*/weights/best.pt")
    ap.add_argument("--source", required=True, help="图片/文件夹/视频路径")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights.strip().strip('"').strip("'")) if args.weights else ROOT / "weights" / "best.pt"
    if not weights.exists():
        weights = latest_best()
    if not weights or not Path(weights).exists():
        print("找不到训练好的权重。请先训练：python train.py")
        return 2

    # 兼容“把文件拖进命令行”时路径带引号的情况
    source = Path(args.source.strip().strip('"').strip("'"))
    if not source.exists():
        print("找不到输入：", source)
        return 2

    print("模型:", weights)
    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(ROOT / args.out),
        name="predict",
        exist_ok=True,
        verbose=False,
    )

    names = model.names
    total = 0
    for r in results:
        print(f"\n[{Path(r.path).name}]")
        if r.boxes is None or len(r.boxes) == 0:
            print("  未检测到目标")
            continue
        for box in r.boxes:
            cid = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            total += 1
            print(f"  {names.get(cid, cid):10s} conf={conf:.2f}  box={xyxy}")
    print(f"\n共检测到 {total} 个目标，结果图保存在 {ROOT / args.out / 'predict'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
