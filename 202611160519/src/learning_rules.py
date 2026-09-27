# learning_rules.py
# 学习规则层：禁区 + 标签修正 + 漏检收敛点
# 不直接改 GT，独立存储；训练导出 / 推理时应用
import json
from datetime import datetime
from pathlib import Path

from core import compute_iou


# 禁区默认参数
DEFAULT_FORBIDDEN_RATIO = 0.02   # 半径 = min(iw,ih) * ratio
MIN_FORBIDDEN_RADIUS = 6         # 最小半径（低于这个放弃）
MAX_FORBIDDEN_RADIUS = 40        # 最大半径硬上限
FORBIDDEN_MARGIN = 3             # 距离 GT 顶点至少留 3px
DEFAULT_MISS_REPEAT = 8          # 漏检图在训练集里复制多少份


def _dist2(ax, ay, bx, by):
    dx = ax - bx; dy = ay - by
    return dx * dx + dy * dy


def _min_dist_to_gt_corners(cx, cy, gt_boxes):
    """点到所有 GT 框 4 顶点的最小距离"""
    best = float("inf")
    if not gt_boxes:
        return best
    for g in gt_boxes:
        x1, y1, x2, y2 = g["xyxy"]
        for (gx, gy) in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            d = _dist2(cx, cy, gx, gy) ** 0.5
            if d < best:
                best = d
    return best


def _compute_wanted_radius(box_xyxy, img_size):
    """按图大小算期望半径，并受框边长限制"""
    x1, y1, x2, y2 = box_xyxy
    bw = abs(x2 - x1); bh = abs(y2 - y1)
    min_edge = max(min(bw, bh), 1.0)
    if img_size:
        iw, ih = img_size
        base = min(iw, ih) * DEFAULT_FORBIDDEN_RATIO
    else:
        base = 15.0
    # 上限 1：全局
    r = min(base, MAX_FORBIDDEN_RADIUS)
    # 上限 2：不超过短边的 1/4，避免 4 个圆互相重叠
    r = min(r, min_edge / 4.0)
    return max(r, 0.0)


def _safe_radius(cx, cy, gt_boxes, r_wanted,
                 margin=FORBIDDEN_MARGIN, min_r=MIN_FORBIDDEN_RADIUS):
    """
    安全半径：必须比到最近 GT 顶点的距离小 margin。
    距离太近返回 None 表示放弃。
    """
    if not gt_boxes:
        return r_wanted if r_wanted >= min_r else None
    min_d = _min_dist_to_gt_corners(cx, cy, gt_boxes)
    r_safe = min(r_wanted, min_d - margin)
    if r_safe < min_r:
        return None
    return float(r_safe)


class LearningRules:
    """
    三类规则：
      forbidden_zones:   {img_name: [{"cx","cy","r","created"}]}
      label_corrections: {img_name: [{"old_xyxy","new_xyxy",
                                      "new_cls_id","new_cls_name","created"}]}
      miss_points:       {img_name: [{"xyxy","cls_id","cls_name",
                                      "repeat","created"}]}
    """

    def __init__(self, path):
        self.path = Path(path)
        self.data = {
            "forbidden_zones": {},
            "label_corrections": {},
            "miss_points": {},
        }
        self.load()

    # ==================== 存取 ====================
    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                for k in ("forbidden_zones", "label_corrections", "miss_points"):
                    d.setdefault(k, {})
                self.data = d
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _key(name):
        return Path(name).name

    # ==================== 禁区 ====================
    def add_forbidden_from_box(self, img_name, box_xyxy, gt_boxes,
                               img_size=None):
        """
        以框的 4 个顶点为圆心生成禁区圆。
        - gt_boxes: 该图当前的正确框，用于避免圈住答案顶点
        - img_size: (iw, ih) 可选，按图缩放半径
        返回 (added, skipped)
        """
        x1, y1, x2, y2 = box_xyxy
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        r_wanted = _compute_wanted_radius(box_xyxy, img_size)

        k = self._key(img_name)
        self.data["forbidden_zones"].setdefault(k, [])
        added = 0
        skipped = 0
        for (cx, cy) in corners:
            r_safe = _safe_radius(cx, cy, gt_boxes, r_wanted)
            if r_safe is None:
                skipped += 1
                continue
            self.data["forbidden_zones"][k].append({
                "cx": float(cx), "cy": float(cy), "r": r_safe,
                "created": datetime.now().isoformat(timespec="seconds"),
            })
            added += 1
        self.save()
        return added, skipped

    def add_forbidden_zone(self, img_name, cx, cy, gt_boxes,
                           img_size=None, box_xyxy=None):
        """单点禁区；如果传了 box_xyxy 会按框大小算半径"""
        if box_xyxy is not None:
            r_wanted = _compute_wanted_radius(box_xyxy, img_size)
        elif img_size:
            iw, ih = img_size
            r_wanted = min(min(iw, ih) * DEFAULT_FORBIDDEN_RATIO,
                           MAX_FORBIDDEN_RADIUS)
        else:
            r_wanted = 15.0
        r_safe = _safe_radius(cx, cy, gt_boxes, r_wanted)
        if r_safe is None:
            return False
        k = self._key(img_name)
        self.data["forbidden_zones"].setdefault(k, []).append({
            "cx": float(cx), "cy": float(cy), "r": r_safe,
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        self.save()
        return True

    def is_point_in_forbidden(self, img_name, x, y):
        zones = self.data["forbidden_zones"].get(self._key(img_name), [])
        for z in zones:
            if _dist2(x, y, z["cx"], z["cy"]) <= z["r"] * z["r"]:
                return True
        return False

    def _box_hits_zone(self, zones, b):
        x1, y1, x2, y2 = b["xyxy"]
        for (cx, cy) in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            for z in zones:
                if _dist2(cx, cy, z["cx"], z["cy"]) <= z["r"] * z["r"]:
                    return True
        return False

    def filter_boxes_by_forbidden(self, img_name, boxes, gt_boxes=None,
                                  iou_thr=0.5, cls_aware=True):
        """
        过滤落在禁区里的框。
        - gt_boxes 提供了的话：和任意 GT 位置匹配（可选类别相同）的框豁免。
        - cls_aware=True: 类别也必须相同才豁免。
        """
        zones = self.data["forbidden_zones"].get(self._key(img_name), [])
        if not zones:
            return list(boxes)

        def _matched_gt(b):
            if not gt_boxes:
                return False
            for g in gt_boxes:
                if cls_aware and int(g.get("cls_id", -1)) != int(b.get("cls_id", -2)):
                    continue
                if compute_iou(b["xyxy"], g["xyxy"]) >= iou_thr:
                    return True
            return False

        out = []
        for b in boxes:
            if self._box_hits_zone(zones, b) and not _matched_gt(b):
                continue
            out.append(b)
        return out

    def get_forbidden_zones(self, img_name):
        return list(self.data["forbidden_zones"].get(self._key(img_name), []))

    def remove_last_forbidden(self, img_name):
        k = self._key(img_name)
        if k in self.data["forbidden_zones"] and self.data["forbidden_zones"][k]:
            self.data["forbidden_zones"][k].pop()
            self.save()
            return True
        return False

    def clear_forbidden(self, img_name=None):
        if img_name is None:
            self.data["forbidden_zones"] = {}
        else:
            self.data["forbidden_zones"].pop(self._key(img_name), None)
        self.save()

    # ==================== 标签修正 ====================
    def add_label_correction(self, img_name, old_xyxy, new_xyxy,
                             new_cls_id, new_cls_name):
        k = self._key(img_name)
        self.data["label_corrections"].setdefault(k, []).append({
            "old_xyxy": [float(v) for v in old_xyxy],
            "new_xyxy": [float(v) for v in new_xyxy],
            "new_cls_id": int(new_cls_id),
            "new_cls_name": str(new_cls_name),
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        self.save()

    def apply_corrections(self, img_name, boxes, iou_thr=0.5):
        corrs = self.data["label_corrections"].get(self._key(img_name), [])
        if not corrs:
            return list(boxes)
        out = []
        for b in boxes:
            matched = None
            for c in corrs:
                if compute_iou(b["xyxy"], c["old_xyxy"]) > iou_thr:
                    matched = c
                    break
            if matched:
                out.append({
                    "xyxy": list(matched["new_xyxy"]),
                    "cls_id": int(matched["new_cls_id"]),
                    "cls_name": str(matched["new_cls_name"]),
                    "conf": b.get("conf", 1.0),
                    "manual": False,
                    "required": b.get("required", False),
                })
            else:
                out.append(b)
        return out

    def get_corrections(self, img_name):
        return list(self.data["label_corrections"].get(self._key(img_name), []))

    def clear_corrections(self, img_name=None):
        if img_name is None:
            self.data["label_corrections"] = {}
        else:
            self.data["label_corrections"].pop(self._key(img_name), None)
        self.save()

    # ==================== 漏检收敛点 ====================
    def add_miss_point(self, img_name, xyxy, cls_id, cls_name,
                       repeat=DEFAULT_MISS_REPEAT):
        k = self._key(img_name)
        self.data["miss_points"].setdefault(k, []).append({
            "xyxy": [float(v) for v in xyxy],
            "cls_id": int(cls_id),
            "cls_name": str(cls_name),
            "repeat": int(repeat),
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        self.save()

    def get_miss_points(self, img_name):
        return list(self.data["miss_points"].get(self._key(img_name), []))

    def clear_miss_points(self, img_name=None):
        if img_name is None:
            self.data["miss_points"] = {}
        else:
            self.data["miss_points"].pop(self._key(img_name), None)
        self.save()

    # ==================== 汇总 ====================
    def has_any(self, img_name=None):
        if img_name is None:
            return self.count() > 0
        k = self._key(img_name)
        return (k in self.data["forbidden_zones"] or
                k in self.data["label_corrections"] or
                k in self.data["miss_points"])

    def count_forbidden(self):
        return sum(len(v) for v in self.data["forbidden_zones"].values())

    def count_corrections(self):
        return sum(len(v) for v in self.data["label_corrections"].values())

    def count_miss_points(self):
        return sum(len(v) for v in self.data["miss_points"].values())

    def count(self):
        return (self.count_forbidden() + self.count_corrections()
                + self.count_miss_points())

    def all_names(self):
        keys = set()
        keys |= set(self.data["forbidden_zones"].keys())
        keys |= set(self.data["label_corrections"].keys())
        keys |= set(self.data["miss_points"].keys())
        return keys

    def clear_all(self):
        self.data = {
            "forbidden_zones": {},
            "label_corrections": {},
            "miss_points": {},
        }
        self.save()