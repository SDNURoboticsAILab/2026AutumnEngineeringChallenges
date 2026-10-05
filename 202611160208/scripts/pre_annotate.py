# -*- coding: utf-8 -*-
"""YOLO-World 开放词汇预标注（Level 2）：用文字提示词对全部图片生成初始 YOLO 标签，
之后人工逐张修正，节省纯手工标注时间。

用法: python scripts/pre_annotate.py [--conf 0.25]
输出: yolo_project/dataset/labels/{train,val}/<同名>.txt
"""
import argparse
import os

from ultralytics import YOLOWorld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")

# 提示词 → 类别编号（与 data.yaml 一致；"obstacle"抽象词检不出，改用具体外观词）
# obstacle 定义（实地看图确认）：画面中心的蓝色泡沫盒子，椅子/门架等背景杂物不标注
PROMPTS = [
    ("blue box", 0),
    ("cyan box", 0),
    ("teal box", 0),
    ("blue crate", 0),
    ("foam box", 0),
    ("coke bottle", 1),
    ("dark bottle", 1),
    ("plastic bottle", 1),
    ("soccer ball", 2),
]
CLASSES = [p for p, _ in PROMPTS]
CLASS_IDS = [c for _, c in PROMPTS]


def iou(a, b):
    """两个 [cx,cy,w,h] 归一化框的 IoU"""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0


def dedup(boxes, confs, ids, thr=0.5):
    """同类别内 IoU>thr 的框只保留置信度最高者（多提示词会重复框同一物体）"""
    order = sorted(range(len(boxes)), key=lambda k: confs[k], reverse=True)
    keep = []
    for idx in order:
        if all(not (ids[idx] == ids[j] and iou(boxes[idx], boxes[j]) > thr) for j in keep):
            keep.append(idx)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = ap.parse_args()

    # GitHub 直连被重置，权重已手动下载至 downloads/；若不存在则回退自动下载
    weight = os.path.join(ROOT, "downloads", "yolov8s-worldv2.pt")
    model = YOLOWorld(weight if os.path.exists(weight) else "yolov8s-worldv2.pt")
    model.set_classes(CLASSES)

    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET, "images", split)
        lbl_dir = os.path.join(DATASET, "labels", split)
        os.makedirs(lbl_dir, exist_ok=True)
        files = [f for f in sorted(os.listdir(img_dir))
                 if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
        print(f"[{split}] 共 {len(files)} 张，开始预标注…")

        n_empty = 0
        for i, f in enumerate(files, 1):
            img_path = os.path.join(img_dir, f)
            results = model.predict(img_path, conf=args.conf, verbose=False)
            r = results[0]
            boxes = r.boxes.xywhn.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            raw_cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
            ids = [CLASS_IDS[c] for c in raw_cls]
            keep = dedup(boxes, confs, ids)
            lines = []
            for idx in keep:
                cx, cy, w, h = boxes[idx]
                lines.append(f"{ids[idx]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stem = os.path.splitext(f)[0]
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as fp:
                fp.write("\n".join(lines))
            if not lines:
                n_empty += 1
            if i % 100 == 0 or i == len(files):
                print(f"  进度 {i}/{len(files)}")
        print(f"[{split}] 完成，其中 {n_empty} 张未检出任何目标（需人工补标）")


if __name__ == "__main__":
    main()
