# -*- coding: utf-8 -*-
"""标注质量统计：各类别框数、空标签图片清单（诊断预标注效果）"""
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")
NAMES = {0: "obstacle", 1: "cola", 2: "football"}

for split in ["train", "val"]:
    lbl = os.path.join(DATASET, "labels", split)
    box_cnt = Counter()
    empty = []
    total = 0
    for f in os.listdir(lbl):
        if f == "classes.txt":
            continue
        total += 1
        with open(os.path.join(lbl, f)) as fp:
            lines = [l for l in fp.read().splitlines() if l.strip()]
        if not lines:
            empty.append(f)
        for l in lines:
            box_cnt[NAMES[int(l.split()[0])]] += 1
    print(f"[{split}] 标签文件 {total} 个 | 空 {len(empty)} 个 | 各类框数 {dict(box_cnt)}")
