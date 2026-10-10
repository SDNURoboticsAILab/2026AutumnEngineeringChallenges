"""半自动预标注流水线 —— 为 obstacle / cola / football 生成 YOLO 标签。

为什么这么做（三类目标各用一种最可靠的信号，全部可解释、可复核、可调参）：

  1) obstacle —— 蓝色泡沫箱
     颜色极其纯净（青蓝色），直接用 HSV 阈值分割 + 连通域就能框准，
     完全不依赖检测器，是三类里最稳的。

  2) cola —— 黑色可乐瓶
     试过用 COCO 预训练模型的 bottle 类，实测**完全检不出**（这批图里瓶子太小、
     太暗，COCO 的"瓶子"先验迁移不过来），所以改为几何方案：
        低亮度 + **低饱和度**（纯黑）  →  细高形状  →  独立立在画面中下部
     关键在"低饱和度"这一条：木门架在阴影里同样是"暗色细高"，
     但它是暗棕色（有色相、有饱和度），加上这一条就能把两者分开。

  3) football —— 黑白足球
     COCO 的 sports ball 类召回很好，但场景里有橙色瑜伽球（同样是"球"），
     所以叠加颜色仲裁：框内橙色占比高 → 判为干扰球丢弃；
     框内黑白占比够高 → 判为 football。

用法：
    python tools/pre_annotate.py --weights ../weights/yolov8s.pt --dataset dataset
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

# ---------------------------------------------------------------- 类别定义
CLS_OBSTACLE, CLS_COLA, CLS_FOOTBALL = 0, 1, 2
NAMES = {CLS_OBSTACLE: "obstacle", CLS_COLA: "cola", CLS_FOOTBALL: "football"}

COCO_BOTTLE = 39               # COCO 里"瓶子"的类别下标
COCO_SPORTS_BALL = 32          # COCO 里"球"的类别下标


# ---------------------------------------------------------------- 通用工具
def hsv_split(bgr: np.ndarray):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]


def ratio(bgr: np.ndarray, kind: str) -> float:
    """返回某种颜色在区域中的像素占比。"""
    if bgr is None or bgr.size == 0:
        return 0.0
    h, s, v = hsv_split(bgr)
    if kind == "orange":                       # 橙色瑜伽球
        m = (h >= 5) & (h <= 25) & (s > 90) & (v > 90)
    elif kind == "achromatic_dark":            # 纯黑（可乐瓶瓶身）
        m = (v < 75) & (s < 90)
    elif kind == "blue":                       # 蓝色泡沫箱
        m = (h >= 85) & (h <= 115) & (s > 60) & (v > 80)
    elif kind == "white":                      # 足球的白色块
        m = (s < 60) & (v > 140)
    elif kind == "black":                      # 足球的黑色块
        m = v < 70
    else:
        raise ValueError(kind)
    return float(m.mean())


def mean_value(bgr: np.ndarray) -> float:
    """区域的平均亮度。足球是亮物体，可乐瓶是暗物体，用这个区分最直接。"""
    if bgr is None or bgr.size == 0:
        return 0.0
    return float(hsv_split(bgr)[2].mean())


def components(mask: np.ndarray, open_k: int = 3, close_k: int = 3, close_it: int = 2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=close_it)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [tuple(int(v) for v in stats[i]) for i in range(1, n)]   # x, y, w, h, area


def crop(image: np.ndarray, box_xyxy) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), np.uint8)
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------- obstacle
def detect_obstacles(image, args):
    """蓝色泡沫箱：HSV 蓝色分割 + 连通域 + 形状/纯度过滤。"""
    h, s, v = hsv_split(image)
    mask = (((h >= args.blue_h[0]) & (h <= args.blue_h[1]) &
             (s > args.blue_s) & (v > args.blue_v))).astype(np.uint8) * 255
    out = []
    for x, y, w, hh, area in components(mask, 5, 5, 2):
        if area < args.blue_min_area:
            continue
        if area / float(w * hh) < args.blue_min_fill:
            continue
        if not (0.2 <= w / float(hh) <= 5.0):
            continue
        if ratio(image[y:y + hh, x:x + w], "blue") < args.blue_min_ratio:
            continue
        out.append((CLS_OBSTACLE, x, y, x + w, y + hh))
    return out


# ---------------------------------------------------------------- cola
def detect_cola(model, image, args):
    """黑色可乐瓶：COCO 预训练检测器的 bottle 类 + 暗色校验。

    这里踩的坑是**推理分辨率**，值得单独记一笔：

        最初用 imgsz=640 跑 COCO 的 bottle 类，实测**完全检不出**可乐瓶，于是误判为
        "COCO 的先验迁移不过来"，转去手写颜色+形状规则，连续迭代四版都不理想
        （全局阈值卡不准、门架立柱误检、瓶子高光被切碎……）。

        后来把同一张图分别用 640 和 1280 推理做对照，才发现真正的原因是**分辨率**：
        这批图里可乐瓶只占画面很小一块，640 输入下经过下采样后几乎消失；
        提到 1280 后，6 张抽样图里的命中数从 2 张升到 4 张，置信度最高到 0.89。

    结论：小目标检测要先把输入分辨率这一维试出来，再去怀疑模型能力。
    最终方案 = COCO bottle @1280（保证召回） + 区域必须偏暗（保证精度，剔除误检）。
    """
    if model is None:
        return []
    res = model.predict(image, conf=args.cola_conf, imgsz=args.cola_imgsz, verbose=False,
                        classes=[COCO_BOTTLE])[0]
    out = []
    for box in res.boxes:
        x1, y1, x2, y2 = [float(t) for t in box.xyxy[0].tolist()]
        region = crop(image, (x1, y1, x2, y2))
        if region.size == 0:
            continue
        if mean_value(region) > args.cola_max_mean_v:            # 不够暗 → 不是可乐瓶
            continue
        if ratio(region, "achromatic_dark") < args.cola_min_dark_ratio:
            continue
        out.append((CLS_COLA, x1, y1, x2, y2))
    return out


def detect_cola_geometric(image, args):
    """可乐瓶的几何兜底方案（默认关闭）。

    暗色掩膜 + 竖直方向形态学开运算 + 细高形状约束。思路是"瓶子竖直细长、阴影弥散"，
    用 1×N 的竖条核做开运算可以抹掉弥散阴影、留下竖直瓶身。
    实测召回不如 COCO @1280，仅作为检测器漏检时的补充手段保留。
    """
    h, s, v = hsv_split(image)
    H, W = image.shape[:2]

    mask = ((v < args.cola_v_max) & (s < args.cola_s_max)).astype(np.uint8) * 255
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, args.cola_v_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

    out = []
    for x, y, w, hh, area in components(mask, 1, 3, 1):
        if area < args.cola_min_area or hh < args.cola_min_h or w < args.cola_min_w:
            continue
        if not (args.cola_min_ar <= hh / float(w) <= args.cola_max_ar):
            continue
        if area / float(w * hh) < args.cola_min_fill:
            continue
        if w > W * args.cola_max_w_frac or hh > H * args.cola_max_h_frac:
            continue
        if y <= args.cola_border:                                # 只排除贴上边缘的
            continue
        if (y + hh / 2) / H < args.cola_min_cy:
            continue
        region = image[y:y + hh, x:x + w]
        if mean_value(region) > args.cola_max_mean_v:
            continue
        if ratio(region, "achromatic_dark") < args.cola_min_dark_ratio:
            continue
        out.append((CLS_COLA, x, y, x + w, y + hh))
    return out


# ---------------------------------------------------------------- football
def detect_football(model, image, args):
    """黑白足球：COCO 的 sports ball 类 **多尺度推理** + 颜色仲裁。

    这里得到的是本项目最有价值的一条实测结论：**同一类别在不同尺度下需要不同的推理分辨率。**

        把"漏检的足球图"和"命中了的足球图"分别渲染出来对比，发现两组画面的差别很明显：
            - 漏检组：球都很大很近（画面底部的特写）
            - 命中组：球都很小很远
        COCO 训练集里的 sports ball 通常只占几十像素，所以近景大球**超出了模型的尺度分布**，
        反而检不出。实测各分辨率的命中率（各抽 10 张）：

            imgsz    漏检组(近景大球)   命中组(远景小球)
            640          1/10              10/10
            480          4/10               9/10
            320          4/10               7/10

        两组的最优分辨率正好相反。因此这里改成**多尺度推理再合并**：
        在 640 / 480 / 320 三个尺度各跑一遍，把结果按置信度做 NMS 合并，
        既保住远景小球，又救回近景大球。
    """
    if model is None:
        return []
    cands = []
    for sz in args.ball_imgsz:
        res = model.predict(image, conf=args.conf, imgsz=sz, verbose=False,
                            classes=[COCO_SPORTS_BALL])[0]
        for box in res.boxes:
            x1, y1, x2, y2 = [float(t) for t in box.xyxy[0].tolist()]
            region = crop(image, (x1, y1, x2, y2))
            if region.size == 0:
                continue
            if ratio(region, "orange") > args.orange_reject:    # 橙色瑜伽球，丢弃
                continue
            # 足球是"黑白相间的亮物体"：既要有白色块，整体又不能是暗的。
            # 只判黑白占比会把纯黑的物体（可乐瓶）也算进来，所以必须加亮度这一条。
            if mean_value(region) < args.ball_min_v:
                continue
            if ratio(region, "white") < args.ball_min_white:
                continue
            cands.append((CLS_FOOTBALL, x1, y1, x2, y2, float(box.conf.item())))

    # 跨尺度合并：同一目标只保留置信度最高的那个框
    kept = []
    for d in sorted(cands, key=lambda t: -t[5]):
        if any(k[0] == d[0] and iou(d[1:5], k[1:5]) > args.merge_iou for k in kept):
            continue
        kept.append(d)
    return [(c, x1, y1, x2, y2) for c, x1, y1, x2, y2, _ in kept]


# ---------------------------------------------------------------- 后处理
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def dedup(dets, thr: float):
    kept = []
    for cls, x1, y1, x2, y2 in sorted(dets, key=lambda d: -(d[3] - d[1]) * (d[4] - d[2])):
        if any(k[0] == cls and iou((x1, y1, x2, y2), k[1:]) > thr for k in kept):
            continue
        kept.append((cls, x1, y1, x2, y2))
    return kept


def to_yolo(dets, w: int, h: int):
    lines = []
    for cls, x1, y1, x2, y2 in dets:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0.004 or bh <= 0.004:
            continue
        lines.append(f"{cls} {((x1 + x2) / 2 / w):.6f} {((y1 + y2) / 2 / h):.6f} {bw:.6f} {bh:.6f}")
    return lines


def annotate_image(model, image, args):
    dets = []
    if args.enable_obstacle:
        dets += detect_obstacles(image, args)
    if args.enable_cola:
        dets += detect_cola(model, image, args)
    if args.enable_cola_geometric:
        dets += detect_cola_geometric(image, args)
    if args.enable_football:
        dets += detect_football(model, image, args)
    return dedup(dets, args.iou_dedup)


# ---------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description="生成 obstacle/cola/football 的预标注")
    ap.add_argument("--weights", default="", help="COCO 预训练权重（用于 cola / football）")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--conf", type=float, default=0.15, help="football 检测置信度阈值")
    ap.add_argument("--iou-dedup", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只处理前 N 张（调试用）")
    ap.add_argument("--render", default="", help="把结果画成拼图存到该路径，便于人工复核")

    g = ap.add_argument_group("obstacle（蓝色泡沫箱）")
    g.add_argument("--enable-obstacle", action="store_true", default=True)
    g.add_argument("--blue-h", type=int, nargs=2, default=[85, 115])
    g.add_argument("--blue-s", type=int, default=60)
    g.add_argument("--blue-v", type=int, default=80)
    g.add_argument("--blue-min-area", type=int, default=400)
    g.add_argument("--blue-min-fill", type=float, default=0.45)
    g.add_argument("--blue-min-ratio", type=float, default=0.35)

    c = ap.add_argument_group("cola（黑色可乐瓶）")
    c.add_argument("--enable-cola", action="store_true", default=True)
    c.add_argument("--cola-conf", type=float, default=0.12, help="bottle 检测置信度阈值")
    c.add_argument("--cola-imgsz", type=int, default=1280, help="bottle 检测分辨率")
    c.add_argument("--cola-max-mean-v", type=float, default=130.0, help="区域平均亮度上限")
    c.add_argument("--cola-min-dark-ratio", type=float, default=0.15, help="区域内纯黑像素占比下限")
    c.add_argument("--enable-cola-geometric", action="store_true", default=False,
                   help="额外启用几何兜底方案（默认关闭）")
    c.add_argument("--cola-v-max", type=int, default=135)
    c.add_argument("--cola-s-max", type=int, default=95)
    c.add_argument("--cola-v-kernel", type=int, default=21, help="竖直开运算核高度，抹掉弥散阴影")
    c.add_argument("--cola-min-area", type=int, default=200)
    c.add_argument("--cola-min-w", type=int, default=10)
    c.add_argument("--cola-min-h", type=int, default=45)
    c.add_argument("--cola-min-ar", type=float, default=1.5)
    c.add_argument("--cola-max-ar", type=float, default=8.0)
    c.add_argument("--cola-min-fill", type=float, default=0.30)
    c.add_argument("--cola-max-w-frac", type=float, default=0.30)
    c.add_argument("--cola-max-h-frac", type=float, default=0.90)
    c.add_argument("--cola-border", type=int, default=2)
    c.add_argument("--cola-min-cy", type=float, default=0.25)

    f = ap.add_argument_group("football（黑白足球）")
    f.add_argument("--enable-football", action="store_true", default=True)
    f.add_argument("--ball-imgsz", type=int, nargs="+", default=[640, 480, 320],
                   help="足球的多尺度推理分辨率；远景小球靠 640，近景大球靠 480/320")
    f.add_argument("--merge-iou", type=float, default=0.5, help="跨尺度合并时的 IoU 阈值")
    f.add_argument("--orange-reject", type=float, default=0.25, help="框内橙色占比超过该值判为干扰球")
    f.add_argument("--ball-min-v", type=float, default=95.0, help="框内平均亮度下限：足球是亮物体")
    f.add_argument("--ball-min-white", type=float, default=0.15, help="框内白色占比下限")
    return ap


COLORS = {CLS_OBSTACLE: (255, 170, 40), CLS_COLA: (60, 60, 230), CLS_FOOTBALL: (60, 220, 60)}


def render_sheet(pairs, out_path, cols=4, tile=320):
    rows = (len(pairs) + cols - 1) // cols
    sheet = np.full((rows * tile, cols * tile, 3), 30, np.uint8)
    for i, (img, dets, tag) in enumerate(pairs):
        vis = cv2.resize(img, (tile, tile))
        sx, sy = tile / img.shape[1], tile / img.shape[0]
        for cls, x1, y1, x2, y2 in dets:
            cv2.rectangle(vis, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)),
                          COLORS[cls], 2)
        cv2.putText(vis, tag, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = vis
    cv2.imwrite(out_path, sheet)
    print(f"复核图已保存：{out_path}")


def main() -> int:
    args = build_parser().parse_args()

    model = None
    if args.weights and args.enable_football:
        from ultralytics import YOLO
        model = YOLO(args.weights)
        print(f"已加载 COCO 预训练权重（仅用于 football）：{args.weights}")

    rows, counts, empty, sheet_items = [], defaultdict(int), [], []
    for split in ("train", "val"):
        img_dir = os.path.join(args.dataset, "images", split)
        lbl_dir = os.path.join(args.dataset, "labels", split)
        os.makedirs(lbl_dir, exist_ok=True)
        names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))
        if args.limit:
            names = names[:args.limit]
        for i, name in enumerate(names, 1):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None:
                print(f"[WARN] 无法读取 {name}", file=sys.stderr)
                continue
            h, w = img.shape[:2]
            dets = annotate_image(model, img, args)
            lines = to_yolo(dets, w, h)
            with open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            for cls, *_ in dets:
                counts[NAMES[cls]] += 1
            if not lines:
                empty.append(f"{split}/{name}")
            rows.append({"split": split, "image": name, "n_boxes": len(lines),
                         "classes": "|".join(sorted({NAMES[c] for c, *_ in dets}))})
            if args.render:
                sheet_items.append((img, dets, f"{split[:1]}:{name.split('_')[0]}"))
            if i % 100 == 0:
                print(f"  [{split}] {i}/{len(names)}", flush=True)

    with open("pre_annotate_report.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=["split", "image", "n_boxes", "classes"])
        wtr.writeheader()
        wtr.writerows(rows)

    if args.render and sheet_items:
        step = max(1, len(sheet_items) // 24)
        render_sheet(sheet_items[::step][:24], args.render)

    print("\n=== 预标注统计 ===")
    for k in ("obstacle", "cola", "football"):
        print(f"  {k:9s} 框数 = {counts[k]}")
    print(f"  无目标图片 = {len(empty)} 张")
    if empty:
        print("  前 10 张：", ", ".join(empty[:10]))
    print("逐张明细见 pre_annotate_report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
