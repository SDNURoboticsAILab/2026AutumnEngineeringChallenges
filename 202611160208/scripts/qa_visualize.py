# -*- coding: utf-8 -*-
"""标注质检可视化：把 YOLO 标签画回图片，输出到 yolo_project/qa/ 供人工抽查
用法:
  python scripts/qa_visualize.py                 # 每类随机抽 6 张
  python scripts/qa_visualize.py --per 10        # 每类抽 10 张
  python scripts/qa_visualize.py --split val     # 只抽验证集
"""
import argparse
import os
import random

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")
OUT = os.path.join(ROOT, "yolo_project", "qa")
NAMES = {0: "obstacle", 1: "cola", 2: "football"}
COLORS = {0: (0, 165, 255), 1: (0, 255, 0), 2: (255, 80, 0)}  # BGR


def draw(img_dir, lbl_dir, fname):
    img = cv2.imread(os.path.join(img_dir, fname))
    if img is None:
        return None
    h, w = img.shape[:2]
    stem = os.path.splitext(fname)[0]
    lp = os.path.join(lbl_dir, stem + ".txt")
    n = 0
    if os.path.exists(lp):
        for line in open(lp):
            p = line.split()
            if len(p) < 5:
                continue
            cid, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(cid, (0, 0, 255)), 2)
            cv2.putText(img, NAMES.get(cid, str(cid)), (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(cid, (0, 0, 255)), 1)
            n += 1
    return img, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=6, help="每类抽样张数")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    img_dir = os.path.join(DATASET, "images", args.split)
    lbl_dir = os.path.join(DATASET, "labels", args.split)
    files = sorted(os.listdir(img_dir))
    random.seed(7)
    for cls in ["obstacle", "cola", "football"]:
        pool = [f for f in files if f.startswith(cls)]
        for f in random.sample(pool, min(args.per, len(pool))):
            res = draw(img_dir, lbl_dir, f)
            if res:
                img, n = res
                out = os.path.join(OUT, f"{args.split}_{cls}_{n}boxes_{f}")
                cv2.imwrite(out, img)
                print(f"{cls:<10} {n:>3} 框 -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
