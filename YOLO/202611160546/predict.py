"""新图片目标检测（Level 4）—— 用自己训练的权重对训练集之外的图片做推理。

用法：
    python predict.py                          # 处理 new_images/ 下所有图片
    python predict.py --source some.jpg        # 只处理一张
    python predict.py --conf 0.4 --weights runs/exp1/weights/best.pt

输出：
    results/<图片名>.jpg     画好目标框、类别名、置信度的结果图
    results/detections.csv   检测明细（图片、类别、置信度、框坐标）
"""

from __future__ import annotations

import argparse
import csv
import os

# 三类目标的显示颜色（BGR），和 tools/inspect_labels.py 保持一致
COLORS = {0: (255, 170, 40), 1: (60, 60, 230), 2: (60, 220, 60)}
NAMES = {0: "obstacle", 1: "cola", 2: "football"}
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> int:
    ap = argparse.ArgumentParser(description="用训练好的模型检测新图片")
    ap.add_argument("--weights", default="runs/exp1/weights/best.pt")
    ap.add_argument("--source", default="new_images", help="图片或目录")
    ap.add_argument("--out", default="results")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(f"找不到权重文件 {args.weights}，请先运行 train.py")

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.weights)

    if os.path.isdir(args.source):
        files = [os.path.join(args.source, n) for n in sorted(os.listdir(args.source))
                 if n.lower().endswith(EXTS)]
    else:
        files = [args.source]
    if not files:
        raise SystemExit(f"{args.source} 下没有图片")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 读不到 {path}")
            continue
        res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        n = 0
        for box in res.boxes:
            cid = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            color = COLORS.get(cid, (255, 255, 255))
            label = f"{NAMES.get(cid, cid)} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, max(th, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            rows.append({"image": os.path.basename(path), "class": NAMES.get(cid, cid),
                         "confidence": f"{conf:.4f}", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            n += 1

        out_path = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0] + ".jpg")
        cv2.imwrite(out_path, img)
        print(f"{os.path.basename(path):40s} 检出 {n} 个目标 -> {out_path}")

    csv_path = os.path.join(args.out, "detections.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "class", "confidence", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n共 {len(files)} 张图片，{len(rows)} 个目标，明细见 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
