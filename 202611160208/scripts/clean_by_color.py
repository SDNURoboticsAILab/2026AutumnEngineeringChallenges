# -*- coding: utf-8 -*-
"""train 标签颜色审计/清洗：对 obstacle 框做蓝色HSV校验、football 框做橙色校验
不达标者判为误标。默认 dry-run 统计+可视化抽样，--apply 才写盘。
用法: python scripts/clean_by_color.py [--apply] [--split train]
"""
import argparse
import os
import random

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")

# OpenCV HSV: H 0~179。青蓝色盒子 H≈75~110；橙色 H≈5~25
COLOR_RANGE = {
    0: ((75, 50, 60), (110, 255, 255)),   # obstacle 蓝色占比应 >25%
    2: ((5, 100, 80), (25, 255, 255)),    # football 橙色占比 >30% 判为瑜伽球误标
}


def ratio(img, box, lo, hi):
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    x1 = int(max(0, (cx - bw / 2) * w)); x2 = int(min(w, (cx + bw / 2) * w))
    y1 = int(max(0, (cy - bh / 2) * h)); y2 = int(min(h, (cy + bh / 2) * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    roi = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return cv2.inRange(roi, lo, hi).mean() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写盘删除（默认只统计）")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    img_dir = os.path.join(DATASET, "images", args.split)
    lbl_dir = os.path.join(DATASET, "labels", args.split)
    files = sorted(f for f in os.listdir(lbl_dir)
                   if f.endswith(".txt") and f != "classes.txt")

    stat = {0: [0, 0], 2: [0, 0]}  # cls: [bad, total]
    bad_samples = {0: [], 2: []}
    for lf in files:
        stem = os.path.splitext(lf)[0]
        img = cv2.imread(os.path.join(img_dir, stem + ".jpg"))
        if img is None:
            continue
        lp = os.path.join(lbl_dir, lf)
        boxes = []
        for line in open(lp):
            p = line.split()
            if len(p) == 5:
                boxes.append((int(p[0]), *map(float, p[1:])))
        if not boxes:
            continue
        new_boxes = []
        for b in boxes:
            cid = b[0]
            if cid in COLOR_RANGE:
                lo, hi = COLOR_RANGE[cid]
                r = ratio(img, b[1:], lo, hi)
                stat[cid][1] += 1
                bad = (r < 0.25) if cid == 0 else (r > 0.30)
                if bad:
                    stat[cid][0] += 1
                    if len(bad_samples[cid]) < 6:
                        bad_samples[cid].append((stem, r))
                    continue
            new_boxes.append(b)
        if args.apply and len(new_boxes) != len(boxes):
            with open(lp, "w") as fp:
                for cid, cx, cy, w, h in new_boxes:
                    fp.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"[{args.split}] obstacle 框: {stat[0][1]} 个, 非蓝色误标 {stat[0][0]} 个 "
          f"({stat[0][0]/max(1,stat[0][1]):.0%})")
    print(f"[{args.split}] football 框: {stat[2][1]} 个, 橙色瑜伽球误标 {stat[2][0]} 个 "
          f"({stat[2][0]/max(1,stat[2][1]):.0%})")
    print("obstacle 误标抽样:", [(s, round(r,2)) for s, r in bad_samples[0]])
    print("football 残留抽样:", [(s, round(r,2)) for s, r in bad_samples[2]])
    # 可视化误标框
    out = os.path.join(ROOT, "yolo_project", "qa", "color_bad_samples")
    os.makedirs(out, exist_ok=True)
    random.seed(1)
    for cid, tag in [(0, "obstacle_nonblue"), (2, "football_orange")]:
        for stem, r in bad_samples[cid][:4]:
            img = cv2.imread(os.path.join(img_dir, stem + ".jpg"))
            for line in open(os.path.join(lbl_dir, stem + ".txt")):
                p = line.split()
                if len(p) == 5 and int(p[0]) == cid:
                    box = list(map(float, p[1:]))
                    h, w = img.shape[:2]
                    cx, cy, bw, bh = box
                    cv2.rectangle(img, (int((cx-bw/2)*w), int((cy-bh/2)*h)),
                                  (int((cx+bw/2)*w), int((cy+bh/2)*h)), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(out, f"{tag}_{stem}.jpg"), img)
    print(f"误标可视化已存 {out}")


if __name__ == "__main__":
    main()
