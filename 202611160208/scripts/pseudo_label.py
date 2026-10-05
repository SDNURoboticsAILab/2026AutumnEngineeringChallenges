# -*- coding: utf-8 -*-
"""伪标注补漏：用当前最优模型给 train 集补上缺失的标签（治"漏检"）

原理：fixed_v4 的 obstacle 精确率 0.968（val 实测），它的预测高度可信。
让它扫 train 集，把"模型检出了、但标签里没有"的框补写进标签文件，
消除"蓝盒子被标成背景"的压制信号 → 重训后召回率与置信度回升。

安全设计：
  - 默认 dry-run 只统计+出可视化，--apply 才写盘；
  - 写盘前自动备份到 labels_backup_prelabel/；
  - 只动 train，绝不碰 val（给 val 打伪标签=数据泄漏，指标会虚高骗自己）；
  - 与已有同类框 IoU>0.3 视为重复不补；
  - 输出 pseudo_added.csv 清单 + 前 20 张补框可视化供人工抽查。

用法:
  python scripts/pseudo_label.py --dry                # 先看统计
  python scripts/pseudo_label.py                      # 执行补标（自动备份）
  python scripts/pseudo_label.py --classes 0 2 --conf 0.25   # 同时补 obstacle+football
"""
import argparse
import csv
import os
import shutil

import cv2
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")
DEFAULT_MODEL = os.path.join(ROOT, "yolo_project", "runs", "fixed_v4", "weights", "best.pt")
NAMES = {0: "obstacle", 1: "cola", 2: "football"}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw, ih = max(0, min(ax2, bx2)-max(ax1, bx1)), max(0, min(ay2, by2)-max(ay1, by1))
    inter = iw*ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", default="train", choices=["train"])  # 故意不允许 val
    ap.add_argument("--classes", default="0", help="要补的类别编号，空格分隔，如 '0 2'")
    ap.add_argument("--conf", type=float, default=0.20, help="伪标签置信度下限")
    ap.add_argument("--iou-dup", type=float, default=0.3, help="与已有框 IoU 超过此值视为重复")
    ap.add_argument("--apply", action="store_true", help="真正写盘（默认 dry-run）")
    args = ap.parse_args()
    target_cls = set(int(c) for c in args.classes.split())

    img_dir = os.path.join(DATASET, "images", args.split)
    lbl_dir = os.path.join(DATASET, "labels", args.split)
    vis_dir = os.path.join(ROOT, "yolo_project", "qa", "pseudo_samples")
    os.makedirs(vis_dir, exist_ok=True)

    if args.apply:
        bak = os.path.join(DATASET, "labels_backup_prelabel")
        if not os.path.exists(bak):
            shutil.copytree(lbl_dir, bak)
            print(f"已备份标签 → {bak}")

    model = YOLO(args.model)
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    changed, total_added = [], 0

    for i, f in enumerate(files, 1):
        stem = os.path.splitext(f)[0]
        lp = os.path.join(lbl_dir, stem + ".txt")
        boxes = []
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) == 5:
                    boxes.append((int(p[0]), *map(float, p[1:])))

        r = model.predict(os.path.join(img_dir, f), conf=args.conf, verbose=False)[0]
        added = []
        for b, c in zip(r.boxes.xywhn.cpu().numpy(),
                        r.boxes.cls.cpu().numpy().astype(int)):
            if c not in target_cls:
                continue
            dup = any(cid == c and iou(tuple(map(float, ex[1:])), tuple(b)) > args.iou_dup
                      for cid, *ex in boxes)
            if not dup:
                added.append((c, *map(float, b)))
        if not added:
            continue
        changed.append((f, len(added)))
        total_added += len(added)
        if args.apply:
            with open(lp, "w") as fp:
                for cid, cx, cy, w, h in boxes + added:
                    fp.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if len(changed) <= 20:  # 前20张存可视化
            img = cv2.imread(os.path.join(img_dir, f))
            h, w = img.shape[:2]
            for cid, cx, cy, bw, bh in boxes:
                cv2.rectangle(img, (int((cx-bw/2)*w), int((cy-bh/2)*h)),
                              (int((cx+bw/2)*w), int((cy+bh/2)*h)), (150, 150, 150), 1)
            for cid, cx, cy, bw, bh in added:
                cv2.rectangle(img, (int((cx-bw/2)*w), int((cy-bh/2)*h)),
                              (int((cx+bw/2)*w), int((cy+bh/2)*h)), (0, 0, 255), 2)
                cv2.putText(img, "NEW " + NAMES[cid], (int((cx-bw/2)*w), int((cy-bh/2)*h)-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(vis_dir, f), img)
        if i % 100 == 0:
            print(f"进度 {i}/{len(files)}  待补图 {len(changed)} 张 共 {total_added} 框")

    out = os.path.join(ROOT, "yolo_project", f"pseudo_added_{args.split}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fp:
        wcsv = csv.writer(fp)
        wcsv.writerow(["image", "added_boxes"])
        wcsv.writerows(changed)
    print(f"\n{'已写盘' if args.apply else 'dry-run(未写盘)'}: "
          f"{len(changed)} 张图需补 {total_added} 个框")
    print(f"清单: {out}")
    print(f"抽查可视化(前20张, 红框=新增): {vis_dir}")


if __name__ == "__main__":
    main()
