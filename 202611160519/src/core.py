# core.py
# v1.3
# 常量 + 工具函数 + 数据结构 + 数据增强 + 逐框评估（面积加权+必要框）+ 智能画框算法
# 新增：
#   - evaluate_one 拆分 duplicate_extra / true_extra
#   - merge_boxes_union / merge_boxes_wbf / suggest_duplicate_groups
#   - auto_resolve_duplicates（推理后处理）
#   - grabcut_rect（魔棒替代，OpenCV 自带）
#   - fastsam_rect（可选，FastSAM 点击分割）
#   - ManualMergeLog / GuidanceLog
import json, shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ========== 常量 ==========
MAX_REASONABLE = {"cola": 3, "football": 3, "obstacle": 15}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "obstacle", 1: "cola", 2: "football"}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
CLASS_COLORS = ['#FF3838', '#38FF38', '#3838FF', '#FF38FF',
                '#FFFF38', '#38FFFF', '#FF8038', '#8038FF']
COCO_TO_PROJECT = {39: 1, 41: 1, 32: 2, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0}

STATUS_TEXT = {"ok": "✓ 正常", "miss": "✗ 漏检", "conflict": "! 冲突",
               "unknown": "? 待判定", "manual_reject": "× 手动拒绝",
               "error": "× 错误", "over_count": "⚠ 数量异常",
               "pass": "✓ 通过", "fail": "✗ 未通过", "untested": "? 未测",
               "img_missing": "图片缺失", "required_fail": "✗ 必要框漏检"}
STATUS_COLOR = {"ok": "#00a000", "miss": "#d08000", "conflict": "#d00000",
                "unknown": "#666666", "manual_reject": "#d00000",
                "error": "#d00000", "over_count": "#ff8000",
                "pass": "#00a000", "fail": "#d00000", "untested": "#666666",
                "img_missing": "#cc0000", "required_fail": "#cc0000"}

# 表格显示用的 F1 通过线（只用于显示，不影响学习闭环）
PASS_F1_THR = 0.9


# ========== 工具函数 ==========
def parse_weak_label(filename):
    name = Path(filename).stem.lower()
    for cls in ["cola", "football", "obstacle"]:
        if name.startswith(cls + "_"):
            return cls
    return None


def judge_result(weak_label, detections):
    counts = {}
    for d in detections:
        counts[d["cls_name"]] = counts.get(d["cls_name"], 0) + 1
    if any(counts.get(c, 0) > MAX_REASONABLE[c] for c in MAX_REASONABLE):
        return "over_count"
    detected = set(counts.keys()) - {"obstacle"}
    if weak_label is None:
        return "unknown"
    if weak_label == "cola":
        return "ok" if "cola" in detected else "miss"
    if weak_label == "football":
        if "cola" in detected:
            return "conflict"
        return "ok" if "football" in detected else "miss"
    if weak_label == "obstacle":
        return "ok"
    return "unknown"


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def _contain_ratio(a, b):
    """交集 / 较小面积"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    small = min(aa, ab)
    return inter / small if small > 0 else 0.0


# ========== 逐框评估（面积加权 + 必要框 + 重复框拆分） ==========
def evaluate_one(gt_boxes, pred_boxes, iou_thr=0.5, img_size=None,
                 use_area_weight=True, enforce_required=True):
    """
    逐框评估一张图。
    - 面积加权：大目标权重更高
    - 必要框：任何 required=True 的框漏检/类别错 → 整图 F1=0
    - 重复框：extra 拆成 duplicate_extra（同类重叠）和 true_extra（真误检）
    """
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    # 面积权重
    if use_area_weight and img_size:
        iw, ih = img_size
        total_area = max(iw * ih, 1)
        gt_weights = []
        for g in gt_boxes:
            x1, y1, x2, y2 = g["xyxy"]
            a = max(0, x2 - x1) * max(0, y2 - y1)
            w = float(np.sqrt(a / total_area))
            w = max(0.1, min(w, 1.0))
            gt_weights.append(w)
    else:
        gt_weights = [1.0] * n_gt
    avg_gt_weight = sum(gt_weights) / max(len(gt_weights), 1)

    # IoU 矩阵
    ious = [[0.0] * n_pred for _ in range(n_gt)]
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            ious[i][j] = compute_iou(g["xyxy"], p["xyxy"])

    # 贪心匹配
    pairs = []
    for i in range(n_gt):
        for j in range(n_pred):
            if ious[i][j] >= iou_thr:
                pairs.append((ious[i][j], i, j))
    pairs.sort(reverse=True)

    matched_gt = set(); matched_pred = set()
    tp = []; cls_error = []
    for iou, i, j in pairs:
        if i in matched_gt or j in matched_pred: continue
        matched_gt.add(i); matched_pred.add(j)
        g = gt_boxes[i]; p = pred_boxes[j]
        if int(g["cls_id"]) == int(p["cls_id"]):
            tp.append((i, j, iou))
        else:
            cls_error.append((i, j, iou, g["cls_name"], p["cls_name"]))

    missed = [i for i in range(n_gt) if i not in matched_gt]
    raw_extra = [j for j in range(n_pred) if j not in matched_pred]

    # ===== 拆分 duplicate_extra / true_extra =====
    duplicate_extra = []
    true_extra = []
    for j in raw_extra:
        p = pred_boxes[j]
        is_dup = False
        for mj in matched_pred:
            mp = pred_boxes[mj]
            if int(mp["cls_id"]) != int(p["cls_id"]):
                continue
            iou = compute_iou(p["xyxy"], mp["xyxy"])
            contain = _contain_ratio(p["xyxy"], mp["xyxy"])
            if iou >= 0.45 or contain >= 0.80:
                is_dup = True
                break
        if is_dup:
            duplicate_extra.append(j)
        else:
            true_extra.append(j)

    correct = len(tp); wrong_cls = len(cls_error)
    missed_n = len(missed)
    extra_n = len(true_extra)
    dup_n = len(duplicate_extra)

    # 加权
    tp_weight = sum(gt_weights[i] for i, j, iou in tp)
    fn_weight = sum(gt_weights[i] for i in missed) + \
                sum(gt_weights[i] for i, j, iou, gc, pc in cls_error)
    fp_weight = extra_n * avg_gt_weight

    total_gt_w = tp_weight + fn_weight
    total_pred_w = tp_weight + fp_weight

    prec = tp_weight / total_pred_w if total_pred_w > 0 else (1.0 if total_gt_w == 0 else 0.0)
    rec = tp_weight / total_gt_w if total_gt_w > 0 else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # 必要框检查
    required_violated = False
    required_info = []
    if enforce_required:
        tp_gt_ids = set(i for i, j, iou in tp)
        cls_err_gt_ids = set(i for i, j, iou, gc, pc in cls_error)
        for i, g in enumerate(gt_boxes):
            if g.get("required", False):
                if i in tp_gt_ids:
                    required_info.append((i, g["cls_name"], "hit"))
                elif i in cls_err_gt_ids:
                    required_info.append((i, g["cls_name"], "cls_wrong"))
                    required_violated = True
                else:
                    required_info.append((i, g["cls_name"], "missed"))
                    required_violated = True
        if required_violated:
            f1 = 0.0; prec = 0.0; rec = 0.0

    return {
        "tp": tp, "cls_error": cls_error,
        "missed": missed,
        "extra": true_extra,             # 真误检（不含重复框）
        "duplicate_extra": duplicate_extra,
        "f1": f1, "precision": prec, "recall": rec,
        "n_gt": n_gt, "n_pred": n_pred,
        "correct": correct, "wrong_cls": wrong_cls,
        "missed_n": missed_n, "extra_n": extra_n, "dup_n": dup_n,
        "gt_weights": gt_weights,
        "tp_weight": tp_weight, "fn_weight": fn_weight, "fp_weight": fp_weight,
        "required_violated": required_violated,
        "required_info": required_info,
    }


# ========== 去重 ==========
def dedupe_boxes(boxes, iou_thr=0.55, contain_thr=0.80):
    """基于 IoU + 包含度的去重，保留面积最大的"""
    n = len(boxes)
    if n < 2:
        return list(boxes), []

    def area(b):
        x1, y1, x2, y2 = b["xyxy"]
        return max(0, x2 - x1) * max(0, y2 - y1)

    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda x: -area(x[1]))

    keep = []; rejected = []; used = set()
    for orig_i, box in indexed:
        if orig_i in used: continue
        keep.append(box)
        for other_i, other in indexed:
            if other_i in used or other_i == orig_i: continue
            iou = compute_iou(box["xyxy"], other["xyxy"])
            contain = _contain_ratio(box["xyxy"], other["xyxy"])
            if iou > iou_thr or contain > contain_thr:
                used.add(other_i)
                rejected.append(other)
    return keep, rejected


# ========== 合并框 ==========
def merge_boxes_union(boxes):
    """并集外接矩形"""
    if not boxes:
        return None
    x1 = min(b["xyxy"][0] for b in boxes)
    y1 = min(b["xyxy"][1] for b in boxes)
    x2 = max(b["xyxy"][2] for b in boxes)
    y2 = max(b["xyxy"][3] for b in boxes)
    best = max(boxes, key=lambda b: b.get("conf", 1.0))
    return {
        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "conf": float(max(b.get("conf", 1.0) for b in boxes)),
        "cls_id": int(best["cls_id"]),
        "cls_name": best["cls_name"],
        "manual": True,
        "required": any(b.get("required", False) for b in boxes),
    }


def merge_boxes_wbf(boxes):
    """加权框融合，权重=conf"""
    if not boxes:
        return None
    total = sum(b.get("conf", 1.0) for b in boxes) or 1.0
    x1 = sum(b["xyxy"][0] * b.get("conf", 1.0) for b in boxes) / total
    y1 = sum(b["xyxy"][1] * b.get("conf", 1.0) for b in boxes) / total
    x2 = sum(b["xyxy"][2] * b.get("conf", 1.0) for b in boxes) / total
    y2 = sum(b["xyxy"][3] * b.get("conf", 1.0) for b in boxes) / total
    best = max(boxes, key=lambda b: b.get("conf", 1.0))
    return {
        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "conf": float(best.get("conf", 1.0)),
        "cls_id": int(best["cls_id"]),
        "cls_name": best["cls_name"],
        "manual": True,
        "required": any(b.get("required", False) for b in boxes),
    }


def suggest_duplicate_groups(boxes, iou_thr=0.55, contain_thr=0.80,
                             class_agnostic=False):
    """找出建议合并的框分组，返回 [[idx,...], ...]"""
    n = len(boxes)
    if n < 2:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if not class_agnostic and boxes[i]["cls_id"] != boxes[j]["cls_id"]:
                continue
            iou = compute_iou(boxes[i]["xyxy"], boxes[j]["xyxy"])
            contain = _contain_ratio(boxes[i]["xyxy"], boxes[j]["xyxy"])
            if iou >= iou_thr or contain >= contain_thr:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


# ========== 推理后处理：自动处理重复框 ==========
def auto_resolve_duplicates(preds, iou_thr=0.5, contain_thr=0.8):
    """
    对推理结果做同类重复框自动合并。
    返回 (final_boxes, actions)
    """
    if not preds:
        return [], []

    groups = suggest_duplicate_groups(preds, iou_thr=iou_thr,
                                      contain_thr=contain_thr,
                                      class_agnostic=False)
    if not groups:
        return list(preds), []

    merged_idx = set()
    final = []
    actions = []
    for g in groups:
        merged_idx.update(g)

    # 未参与分组的原样保留
    for i, p in enumerate(preds):
        if i not in merged_idx:
            final.append(p)

    # 每组做 WBF
    for g in groups:
        group_boxes = [preds[i] for i in g]
        merged = merge_boxes_wbf(group_boxes)
        final.append(merged)
        actions.append({"type": "merge", "count": len(g),
                        "merged_xyxy": merged["xyxy"]})
    return final, actions


# ========== 智能画框：GrabCut（魔棒替代） ==========
def grabcut_rect(image_rgb, rough_xyxy, iterations=5, margin=20):
    """
    用 GrabCut 精修一个粗略矩形。
    - rough_xyxy: 用户拖拽的大致框
    - 返回精修后的 (x1,y1,x2,y2)，失败时返回原框
    """
    try:
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in rough_xyxy]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return rough_xyxy

        bx1 = max(0, x1 - margin); by1 = max(0, y1 - margin)
        bx2 = min(w, x2 + margin); by2 = min(h, y2 + margin)

        mask = np.zeros((h, w), np.uint8)
        mask[by1:by2, bx1:bx2] = cv2.GC_PR_BGD
        mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
        cx1 = x1 + (x2 - x1) // 4; cy1 = y1 + (y2 - y1) // 4
        cx2 = x2 - (x2 - x1) // 4; cy2 = y2 - (y2 - y1) // 4
        if cx2 > cx1 and cy2 > cy1:
            mask[cy1:cy2, cx1:cx2] = cv2.GC_FGD

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        img_bgr = image_rgb[:, :, ::-1].copy()

        cv2.grabCut(img_bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)

        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                           1, 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, 8)
        if num <= 1:
            return rough_xyxy
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        nx1 = stats[largest, cv2.CC_STAT_LEFT]
        ny1 = stats[largest, cv2.CC_STAT_TOP]
        nw = stats[largest, cv2.CC_STAT_WIDTH]
        nh = stats[largest, cv2.CC_STAT_HEIGHT]

        if nw * nh < (x2 - x1) * (y2 - y1) * 0.2:
            return rough_xyxy
        if nw * nh > w * h * 0.95:
            return rough_xyxy
        return (float(nx1), float(ny1), float(nx1 + nw), float(ny1 + nh))
    except Exception as e:
        print(f"[GrabCut 失败] {e}")
        return rough_xyxy


# ========== FastSAM（可选） ==========
_FASTSAM_MODEL = None

def fastsam_rect(image_rgb, seed_x, seed_y, model_path="FastSAM-s.pt"):
    """
    用 FastSAM 做点击分割。
    首次调用会加载模型（可能触发下载，由 GUI 层负责询问）。
    返回 (x1, y1, x2, y2) 或 None。
    """
    global _FASTSAM_MODEL
    try:
        from ultralytics import FastSAM
    except Exception as e:
        print(f"[FastSAM] 导入失败: {e}")
        return None
    try:
        if _FASTSAM_MODEL is None:
            _FASTSAM_MODEL = FastSAM(model_path)
        results = _FASTSAM_MODEL.predict(
            image_rgb, device="cpu", retina_masks=True,
            imgsz=1024, conf=0.4, iou=0.9, verbose=False)
        if not results:
            return None
        r = results[0]
        if r.masks is None or len(r.masks.data) == 0:
            return None
        masks = r.masks.data.cpu().numpy()
        h, w = image_rgb.shape[:2]
        # 找包含种子点的 mask
        best = None
        for m in masks:
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
            if 0 <= seed_y < h and 0 <= seed_x < w and m[seed_y, seed_x] > 0:
                area = int(m.sum())
                if best is None or area < best[0]:
                    best = (area, m)
        if best is None:
            return None
        m = best[1].astype(np.uint8)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return None
        return (float(xs.min()), float(ys.min()),
                float(xs.max()), float(ys.max()))
    except Exception as e:
        print(f"[FastSAM 失败] {e}")
        return None


def fastsam_model_exists(model_path="FastSAM-s.pt"):
    """检查 FastSAM 权重是否已在当前目录"""
    return Path(model_path).exists()


# ========== 边缘吸附 ==========
def build_edge_map(img_bgr, canny_low=50, canny_high=150, dilate=2):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    if dilate > 0:
        kernel = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def snap_rect_to_edges(edge_map, x1, y1, x2, y2, search=12, min_hit=3):
    h, w = edge_map.shape[:2]
    if edge_map is None:
        return x1, y1, x2, y2

    def snap_vertical(fixed_x, y_from, y_to):
        y1i = int(max(0, min(y_from, y_to)))
        y2i = int(min(h, max(y_from, y_to)))
        if y2i - y1i < 2: return fixed_x
        best_x, best_score = fixed_x, 0
        for offset in range(0, search + 1):
            for cand in (fixed_x - offset, fixed_x + offset):
                if cand < 0 or cand >= w: continue
                segment = edge_map[y1i:y2i, int(cand)]
                score = int(np.sum(segment > 0))
                if score > best_score:
                    best_score = score; best_x = cand
            if best_score >= min_hit: break
        return best_x if best_score >= min_hit else fixed_x

    def snap_horizontal(fixed_y, x_from, x_to):
        x1i = int(max(0, min(x_from, x_to)))
        x2i = int(min(w, max(x_from, x_to)))
        if x2i - x1i < 2: return fixed_y
        best_y, best_score = fixed_y, 0
        for offset in range(0, search + 1):
            for cand in (fixed_y - offset, fixed_y + offset):
                if cand < 0 or cand >= h: continue
                segment = edge_map[int(cand), x1i:x2i]
                score = int(np.sum(segment > 0))
                if score > best_score:
                    best_score = score; best_y = cand
            if best_score >= min_hit: break
        return best_y if best_score >= min_hit else fixed_y

    nx1 = snap_vertical(int(x1), y1, y2)
    nx2 = snap_vertical(int(x2), y1, y2)
    ny1 = snap_horizontal(int(y1), nx1, nx2)
    ny2 = snap_horizontal(int(y2), nx1, nx2)

    if nx2 <= nx1: nx1, nx2 = x1, x2
    if ny2 <= ny1: ny1, ny2 = y1, y2
    return float(nx1), float(ny1), float(nx2), float(ny2)


# ========== 旧魔棒（保留，GUI 里不再默认调用） ==========
def magic_wand_rect(image_rgb, seed_x, seed_y, tolerance=30, max_area_ratio=0.95):
    """LAB + 形态学 + 最大连通域版本，作为兜底"""
    h, w = image_rgb.shape[:2]
    if not (0 <= seed_x < w and 0 <= seed_y < h):
        return None

    img_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    def _try(tol):
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        lo = (tol,) * 3
        up = (tol,) * 3
        cv2.floodFill(img_lab.copy(), mask, (int(seed_x), int(seed_y)),
                      255, lo, up, flags)
        m = mask[1:-1, 1:-1]
        kernel = np.ones((5, 5), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if num <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x = stats[largest, cv2.CC_STAT_LEFT]
        y = stats[largest, cv2.CC_STAT_TOP]
        ww = stats[largest, cv2.CC_STAT_WIDTH]
        hh = stats[largest, cv2.CC_STAT_HEIGHT]
        area = stats[largest, cv2.CC_STAT_AREA]
        if area < 20:
            return None
        if ww * hh > w * h * max_area_ratio:
            return None
        return (float(x), float(y), float(x + ww), float(y + hh))

    rect = _try(tolerance)
    if rect is None:
        for tol in (tolerance * 2, tolerance * 3):
            rect = _try(tol)
            if rect is not None:
                break
    return rect

# ========== 快速选择（多笔涂抹合并，魔棒替代方案之一） ==========
def quick_select_rect(image_rgb, strokes, tolerance=35, max_area_ratio=0.95):
    """
    快速选择：用户在多处涂抹，把每笔的 floodFill 结果合并，取外接矩形。
    - strokes: [(x1, y1), (x2, y2), ...] 多笔种子点
    - 返回 (x1, y1, x2, y2) 或 None
    """
    h, w = image_rgb.shape[:2]
    if not strokes:
        return None

    combined = np.zeros((h, w), np.uint8)
    img_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    for (sx, sy) in strokes:
        if not (0 <= sx < w and 0 <= sy < h):
            continue
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        lo_diff = (tolerance,) * 3
        up_diff = (tolerance,) * 3
        try:
            cv2.floodFill(img_lab.copy(), mask, (int(sx), int(sy)),
                          255, lo_diff, up_diff, flags)
            combined = np.maximum(combined, mask[1:-1, 1:-1])
        except Exception:
            pass

    ys, xs = np.where(combined > 0)
    if len(xs) == 0:
        return None
    if (xs.max() - xs.min()) * (ys.max() - ys.min()) > w * h * max_area_ratio:
        return None
    return (float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max()))

# ========== 数据结构 ==========
class TextRedirector:
    def __init__(self, widget): self.widget = widget
    def write(self, s):
        s = s.replace("\r", "\n")
        self.widget.config(state="normal")
        self.widget.insert("end", s)
        self.widget.see("end")
        self.widget.config(state="disabled")
    def flush(self): pass


class ErrorBook:
    def __init__(self, path):
        self.path = Path(path); self.data = {}; self.load()

    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception: self.data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def record(self, name, weak, status, dets=None):
        e = self.data.get(name, {"weak_label": weak, "status": status,
            "wrong_count": 0, "last_check": "", "admin_verified": None, "history": []})
        e["weak_label"] = weak; e["status"] = status
        e["last_check"] = datetime.now().isoformat(timespec="seconds")
        if dets is not None:
            e["detections"] = [{"cls_name": d["cls_name"], "conf": d["conf"]} for d in dets]
        if status in ("conflict", "miss", "manual_reject", "over_count", "required_fail"):
            e["wrong_count"] = e.get("wrong_count", 0) + 1
            e["admin_verified"] = None
        elif status == "ok": e["wrong_count"] = 0
        e.setdefault("history", []).append({"time": e["last_check"], "status": status})
        self.data[name] = e; self.save()

    def verify(self, name, correct):
        if name in self.data:
            self.data[name]["admin_verified"] = correct
            if correct: self.data[name]["wrong_count"] = 0
            self.save()

    def remove(self, name):
        if name in self.data: del self.data[name]; self.save()

    def queue(self):
        q = [(k, v) for k, v in self.data.items()
             if v.get("admin_verified") is None and v.get("wrong_count", 0) > 0]
        q.sort(key=lambda x: -x[1].get("wrong_count", 0))
        return q


class GroundTruth:
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.index_path = self.root / "index.json"
        self.img_dir = self.root / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}; self.load()

    def load(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception: self.data = {}

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _key(p): return Path(p).name

    @staticmethod
    def _normalize_boxes(boxes):
        out = []
        for b in boxes:
            out.append({
                "cls_id": int(b["cls_id"]),
                "cls_name": b["cls_name"],
                "xyxy": [float(v) for v in b["xyxy"]],
                "required": bool(b.get("required", False)),
            })
        return out

    def add(self, img_path, boxes, reviewed=False):
        key = self._key(img_path)
        dst = self.img_dir / (Path(img_path).stem + Path(img_path).suffix)
        try:
            if not dst.exists(): shutil.copy2(img_path, dst)
        except Exception: pass
        existing = self.data.get(key, {})
        if "reviewed" not in existing: final_reviewed = reviewed
        else: final_reviewed = existing["reviewed"] or reviewed
        self.data[key] = {
            "file": dst.name,
            "boxes": self._normalize_boxes(boxes),
            "added": existing.get("added", datetime.now().isoformat(timespec="seconds")),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "last_score": existing.get("last_score"),
            "last_eval": existing.get("last_eval"),
            "iterations": existing.get("iterations", 0),
            "status": existing.get("status", "untested"),
            "reviewed": final_reviewed,
        }
        self.save()

    def update_boxes(self, key, boxes, reviewed=True):
        if key not in self.data: return
        self.data[key]["boxes"] = self._normalize_boxes(boxes)
        self.data[key]["updated"] = datetime.now().isoformat(timespec="seconds")
        if reviewed: self.data[key]["reviewed"] = True
        self.save()

    def set_reviewed(self, key, reviewed=True):
        if key not in self.data: return
        self.data[key]["reviewed"] = reviewed; self.save()

    def get_by_key(self, key): return self.data.get(key)
    def all_items(self): return list(self.data.items())
    def all_reviewed_items(self):
        return [(k, v) for k, v in self.data.items() if v.get("reviewed")]

    def remove(self, key):
        if key in self.data:
            r = self.data.pop(key)
            try: (self.img_dir / r["file"]).unlink()
            except Exception: pass
            self.save()

    def update_score(self, key, f1, status):
        if key not in self.data: return
        self.data[key]["last_score"] = float(f1)
        self.data[key]["last_eval"] = datetime.now().isoformat(timespec="seconds")
        self.data[key]["status"] = status
        self.data[key]["iterations"] = self.data[key].get("iterations", 0) + 1
        self.save()

    def count(self): return len(self.data)
    def count_pass(self): return sum(1 for v in self.data.values() if v.get("status") == "pass")
    def count_fail(self): return sum(1 for v in self.data.values() if v.get("status") == "fail")
    def count_reviewed(self): return sum(1 for v in self.data.values() if v.get("reviewed"))
    def count_unreviewed(self): return sum(1 for v in self.data.values() if not v.get("reviewed"))


class ManualMergeLog:
    """记录哪些图发生了人工合并（用于强化训练）"""
    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def record(self, img_name, before_n, after_n, method="union"):
        key = Path(img_name).name
        rec = self.data.get(key, {
            "count": 0, "first": "", "last": "",
            "before_total": 0, "after_total": 0, "methods": []
        })
        ts = datetime.now().isoformat(timespec="seconds")
        if not rec["first"]:
            rec["first"] = ts
        rec["last"] = ts
        rec["count"] += 1
        rec["before_total"] += int(before_n)
        rec["after_total"] += int(after_n)
        rec["methods"].append(method)
        self.data[key] = rec
        self.save()

    def is_merged(self, img_name):
        return Path(img_name).name in self.data

    def get(self, img_name):
        return self.data.get(Path(img_name).name)

    def all_names(self):
        return set(self.data.keys())

    def count(self):
        return len(self.data)


class GuidanceLog:
    """
    人工引导记录：
      - missed:    人工补漏的框
      - wrong_cls: 人工纠正的类别
      - duplicate: 人工标记的重复框合并
    不改 GT，只用于训练和推理覆盖。
    """
    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _key(name): return Path(name).name

    def _rec(self, name):
        k = self._key(name)
        rec = self.data.setdefault(k, {
            "missed": [], "wrong_cls": [], "duplicate": [],
            "count": 0, "last": ""
        })
        return k, rec

    def add_missed(self, img_name, xyxy, cls_id, cls_name):
        k, rec = self._rec(img_name)
        rec["missed"].append({
            "xyxy": [float(v) for v in xyxy],
            "cls_id": int(cls_id), "cls_name": cls_name,
            "added": datetime.now().isoformat(timespec="seconds")})
        rec["count"] += 1
        rec["last"] = datetime.now().isoformat(timespec="seconds")
        self.save()

    def add_wrong_cls(self, img_name, pred_xyxy, from_cls, to_cls):
        k, rec = self._rec(img_name)
        rec["wrong_cls"].append({
            "pred_xyxy": [float(v) for v in pred_xyxy],
            "from_cls": from_cls, "to_cls": to_cls,
            "added": datetime.now().isoformat(timespec="seconds")})
        rec["count"] += 1
        rec["last"] = datetime.now().isoformat(timespec="seconds")
        self.save()

    def add_duplicate(self, img_name, group_xyxy, merged_xyxy, cls_id=0, cls_name="obstacle"):
        k, rec = self._rec(img_name)
        rec["duplicate"].append({
            "group": [[float(v) for v in b] for b in group_xyxy],
            "merged_xyxy": [float(v) for v in merged_xyxy],
            "cls_id": int(cls_id), "cls_name": cls_name,
            "added": datetime.now().isoformat(timespec="seconds")})
        rec["count"] += 1
        rec["last"] = datetime.now().isoformat(timespec="seconds")
        self.save()

    def get(self, img_name):
        return self.data.get(self._key(img_name))

    def has(self, img_name):
        return self._key(img_name) in self.data

    def all_names(self):
        return set(self.data.keys())

    def count(self):
        return len(self.data)

    def remove(self, img_name):
        k = self._key(img_name)
        if k in self.data:
            del self.data[k]
            self.save()

    def clear(self):
        self.data = {}
        self.save()


# ========== 数据增强 ==========
def _read_labels(p):
    out = []
    if not Path(p).exists(): return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try: out.append([int(parts[0])] + [float(x) for x in parts[1:5]])
                except Exception: pass
    return out


def _write_labels(p, boxes):
    with open(p, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def _aug_brightness(img, boxes, d):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[:, :, 2] = np.clip(h[:, :, 2] + d, 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR), [b[:] for b in boxes]


def _aug_contrast(img, boxes, f):
    m = img.mean()
    return np.clip((img.astype(np.float32) - m) * f + m, 0, 255).astype(np.uint8), [b[:] for b in boxes]


def _aug_saturation(img, boxes, f):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[:, :, 1] = np.clip(h[:, :, 1] * f, 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR), [b[:] for b in boxes]


def _aug_hue(img, boxes, d):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[:, :, 0] = (h[:, :, 0] + d) % 180
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR), [b[:] for b in boxes]


def _aug_hflip(img, boxes):
    return cv2.flip(img, 1), [[b[0], 1.0 - b[1], b[2], b[3], b[4]] for b in boxes]


def _aug_noise(img, boxes, s):
    n = np.random.normal(0, s, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8), [b[:] for b in boxes]


def _aug_blur(img, boxes, k):
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k, k), 0), [b[:] for b in boxes]


def _aug_crop(img, boxes, r):
    import random
    h, w = img.shape[:2]
    ch = int(h * r); cw = int(w * r)
    t = random.randint(0, ch); l = random.randint(0, cw)
    img2 = img[t:h - (ch - t), l:w - (cw - l)]
    nh, nw = img2.shape[:2]
    new = []
    for b in boxes:
        c, cx, cy, bw, bh = b
        x1 = (cx - bw/2) * w - l; y1 = (cy - bh/2) * h - t
        x2 = (cx + bw/2) * w - l; y2 = (cy + bh/2) * h - t
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(nw, x2); y2 = min(nh, y2)
        if x2 - x1 < 5 or y2 - y1 < 5: continue
        new.append([c, ((x1+x2)/2)/nw, ((y1+y2)/2)/nh, (x2-x1)/nw, (y2-y1)/nh])
    return img2, new


def get_augmentations():
    import random
    return [
        ("bright_up", lambda i, b: _aug_brightness(i, b, random.randint(20, 50))),
        ("bright_dn", lambda i, b: _aug_brightness(i, b, -random.randint(20, 50))),
        ("contrast", lambda i, b: _aug_contrast(i, b, random.uniform(0.7, 1.3))),
        ("saturation", lambda i, b: _aug_saturation(i, b, random.uniform(0.6, 1.4))),
        ("hue", lambda i, b: _aug_hue(i, b, random.randint(-10, 10))),
        ("hflip", _aug_hflip),
        ("noise", lambda i, b: _aug_noise(i, b, random.uniform(5, 15))),
        ("blur", lambda i, b: _aug_blur(i, b, random.choice([3, 5]))),
        ("crop", lambda i, b: _aug_crop(i, b, random.uniform(0.05, 0.15))),
    ]