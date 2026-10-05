# -*- coding: utf-8 -*-
"""YOLO 推理脚本（Level 4）：加载自训模型，对新图片检测并保存结果

用法:
  python yolo_project/predict.py                  # 默认 fixed_v4 权重，检测 new_images/ 全部图
  python yolo_project/predict.py --source 图.jpg  # 指定单张或目录，--conf 0.4 改阈值

输出: yolo_project/results/<时间戳>/ 带框结果图 + detections.csv
"""
import argparse
import csv
import os
import time

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "yolo_project", "runs", "fixed_v4", "weights", "best.pt")
DEFAULT_SOURCE = os.path.join(ROOT, "new_images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="自训权重路径(best.pt)")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="图片文件或目录")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    args = ap.parse_args()

    model = YOLO(args.model)
    print(f"已加载模型: {args.model}")
    print(f"类别: {model.names}")

    out_dir = os.path.join(ROOT, "yolo_project", "results", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    results = model.predict(args.source, conf=args.conf, iou=args.iou,
                            save=True, project=out_dir, name="detect", verbose=False)

    rows = []
    for r in results:
        names = [os.path.basename(r.path)] * len(r.boxes)
        for b in r.boxes:
            rows.append([names[0], model.names[int(b.cls)], f"{float(b.conf):.3f}"])
    csv_path = os.path.join(out_dir, "detections.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["image", "class", "confidence"])
        w.writerows(rows)

    print(f"\n共检测 {len(results)} 张图，{len(rows)} 个目标")
    print(f"结果图: {os.path.join(out_dir, 'detect')}")
    print(f"明细表: {csv_path}")


if __name__ == "__main__":
    main()
