# -*- coding: utf-8 -*-
"""train 集标签定向清洗（模型辅助）：用 YOLO-World 以 'orange ball'/'soccer goal net'
定位背景干扰物，删除与它们重叠(IoU>0.3 或中心落入)的 football/obstacle 误标框，
其余类别框不动。清洗前已备份 labels。

用法: python scripts/clean_train_labels.py [--dry]   # --dry 只统计不写盘
"""
import argparse
import os

import cv2
from ultralytics import YOLOWorld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")

# 干扰物提示词 → 要清理的目标类别
# 注意: 不能用 "exercise ball"——CLIP 会把黑白足球也匹配成 exercise ball（实测已踩坑），
# 必须用颜色词 "orange ball" 并配合 HSV 橙色像素校验。
NUISANCE = [
    (["orange ball"], 2),          # 瑜伽球被误标 football(2)
    (["soccer goal net"], 0),      # 网球门被误标 obstacle(0)
]


def is_orange(img_bgr, box_norm, thr=0.3):
    """框内像素 HSV 橙色占比校验（H 5~25, S>100, V>80）"""
    h, w = img_bgr.shape[:2]
    cx, cy, bw, bh = box_norm
    x1 = int(max(0, (cx - bw / 2) * w)); x2 = int(min(w, (cx + bw / 2) * w))
    y1 = int(max(0, (cy - bh / 2) * h)); y2 = int(min(h, (cy + bh / 2) * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    roi = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(roi, (5, 100, 80), (25, 255, 255))
    return mask.mean() / 255 > thr


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw, ih = max(0, min(ax2,bx2)-max(ax1,bx1)), max(0, min(ay2,by2)-max(ay1,by1))
    inter = iw*ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0


def inside(b, nuisance):
    """b 的中心落在任一 nuisance 框内"""
    cx, cy = b[0], b[1]
    for n in nuisance:
        if abs(cx-n[0]) < n[2]/2 and abs(cy-n[1]) < n[3]/2:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只统计不写盘")
    args = ap.parse_args()

    weight = os.path.join(ROOT, "downloads", "yolov8s-worldv2.pt")
    model = YOLOWorld(weight)

    stats = {0: 0, 2: 0}
    n_imgs = 0
    for split in ["train"]:
        img_dir = os.path.join(DATASET, "images", split)
        lbl_dir = os.path.join(DATASET, "labels", split)
        files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        for i, f in enumerate(files, 1):
            stem = os.path.splitext(f)[0]
            lp = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lp):
                continue
            boxes = []
            for line in open(lp):
                p = line.split()
                if len(p) == 5:
                    boxes.append((int(p[0]), *map(float, p[1:])))
            if not boxes:
                continue
            n_imgs += 1
            drop = 0
            img_bgr = cv2.imread(os.path.join(img_dir, f))
            for prompts, target_cls in NUISANCE:
                model.set_classes(prompts)
                r = model.predict(os.path.join(img_dir, f), conf=0.3, verbose=False)[0]
                # 对瑜伽球：只保留框内橙色像素占比达标的 nuisance 框（防足球被误匹配）
                nuis = []
                for b in r.boxes.xywhn.cpu().numpy():
                    if target_cls == 2 and not is_orange(img_bgr, b):
                        continue
                    nuis.append(tuple(b))
                keep = []
                for b in boxes:
                    if b[0] == target_cls and any(
                            iou(b[1:], n) > 0.3 or inside(b[1:], [n]) for n in nuis):
                        drop += 1
                        stats[target_cls] += 1
                        continue
                    keep.append(b)
                boxes = keep  # 链式过滤，下一轮基于已清洗结果
            if drop and not args.dry:
                with open(lp, "w") as fp:
                    for cid, cx, cy, w, h in boxes:
                        fp.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            if i % 100 == 0:
                print(f"进度 {i}/{len(files)}  已删 football={stats[2]} obstacle={stats[0]}")
    print(f"\n完成: 扫描 {n_imgs} 张, 删除 football 误标 {stats[2]} 个, "
          f"obstacle 误标 {stats[0]} 个{'（dry-run 未写盘）' if args.dry else ''}")


if __name__ == "__main__":
    main()
