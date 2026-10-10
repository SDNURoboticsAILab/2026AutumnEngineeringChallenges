"""标注复核工具：把 YOLO 标签画回图片并拼成一张大图，方便逐张核对。

用途：
    - 预标注结束后，抽样渲染检查框是否贴合目标、有没有漏标/误标；
    - 训练前确认标签格式无误（坐标是否越界、类别是否正确）。

用法：
    python tools/inspect_labels.py --dataset dataset --split train --n 16 --cls obstacle
"""

from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np

COLORS = {0: (255, 170, 40), 1: (60, 60, 230), 2: (60, 220, 60)}   # BGR
LABELS = {0: "obstacle", 1: "cola", 2: "football"}


def draw(img: np.ndarray, label_path: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    if os.path.exists(label_path):
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                col = COLORS.get(c, (255, 255, 255))
                cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
                cv2.putText(out, LABELS.get(c, str(c)), (x1 + 3, max(14, y1 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=16, help="拼图里放几张")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cls", default="", help="只看含某类目标的图片，如 obstacle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--tile", type=int, default=320)
    args = ap.parse_args()

    img_dir = os.path.join(args.dataset, "images", args.split)
    lbl_dir = os.path.join(args.dataset, "labels", args.split)
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))

    if args.cls:
        want = {k for k, v in LABELS.items() if v == args.cls}
        keep = []
        for n in names:
            lp = os.path.join(lbl_dir, os.path.splitext(n)[0] + ".txt")
            if not os.path.exists(lp):
                continue
            with open(lp, encoding="utf-8") as f:
                cls_in = {int(float(l.split()[0])) for l in f if l.split()}
            if cls_in & want:
                keep.append(n)
        names = keep

    random.Random(args.seed).shuffle(names)
    names = names[:args.n]
    if not names:
        print("没有符合条件的图片")
        return 1

    rows = (len(names) + args.cols - 1) // args.cols
    t = args.tile
    sheet = np.full((rows * t, args.cols * t, 3), 30, np.uint8)
    for i, n in enumerate(names):
        img = cv2.imread(os.path.join(img_dir, n))
        if img is None:
            continue
        img = cv2.resize(img, (t, t))
        img = draw(img, os.path.join(lbl_dir, os.path.splitext(n)[0] + ".txt"))
        r, c = divmod(i, args.cols)
        sheet[r * t:(r + 1) * t, c * t:(c + 1) * t] = img

    out = args.out or f"check_{args.split}_{args.cls or 'all'}.png"
    cv2.imwrite(out, sheet)
    print(f"已生成 {out}（{len(names)} 张）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
