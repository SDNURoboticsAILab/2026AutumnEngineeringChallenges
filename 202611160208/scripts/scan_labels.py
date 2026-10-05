# -*- coding: utf-8 -*-
"""扫描所有 YOLO 标签，找会导致 labelImg YoloReader 崩溃的异常行。
检查：类别号越界(>=3)、字段数!=5、坐标非数字/越界[0,1]、图片文件缺失、标签对应图片缺失。"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")
NCls = 3

problems = 0
for split in ["train", "val"]:
    img_dir = os.path.join(DATASET, "images", split)
    lbl_dir = os.path.join(DATASET, "labels", split)
    for f in sorted(os.listdir(lbl_dir)):
        if not f.endswith(".txt") or f == "classes.txt":
            continue
        stem = os.path.splitext(f)[0]
        # 图片是否存在
        if not os.path.exists(os.path.join(img_dir, stem + ".jpg")):
            print(f"[无对应图片] {split}/{f}")
            problems += 1
        with open(os.path.join(lbl_dir, f)) as fp:
            for ln, line in enumerate(fp, 1):
                s = line.strip()
                if not s:
                    continue
                p = s.split()
                if len(p) != 5:
                    print(f"[字段数{len(p)}] {split}/{f}:{ln}: {s}")
                    problems += 1
                    continue
                try:
                    cid = int(p[0])
                    vals = list(map(float, p[1:5]))
                except ValueError:
                    print(f"[非数字] {split}/{f}:{ln}: {s}")
                    problems += 1
                    continue
                if cid < 0 or cid >= NCls:
                    print(f"[类别越界{cid}] {split}/{f}:{ln}")
                    problems += 1
                if any(v < 0 or v > 1 for v in vals):
                    print(f"[坐标越界] {split}/{f}:{ln}: {s}")
                    problems += 1
print(f"\n共发现 {problems} 处异常")
