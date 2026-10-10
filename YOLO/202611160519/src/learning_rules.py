# learning_rules.py
# 学习规则层：禁区 + 标签修正 + 漏检点 + 偏大/偏小 + 误检收敛
# 不直接改 GT，独立存储；训练导出 / 推理时应用
import json
from datetime import datetime
from pathlib import Path

from core import compute_iou


DEFAULT_FORBIDDEN_RATIO = 0.02
MIN_FORBIDDEN_RADIUS = 6
MAX_FORBIDDEN_RADIUS = 40
FORBIDDEN_MARGIN = 3
DEFAULT_MISS_REPEAT = 8


def _dist2(ax, ay, bx, by):
    dx = ax - bx; dy = ay - by
    return dx * dx + dy * dy


def _min_dist_to_gt_corners(cx, cy, gt_boxes):
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
    x1, y1, x2, y2 = box_xyxy
    bw = abs(x2 - x1); bh = abs(y2 - y1)
    min_edge = max(min(bw, bh), 1.0)
    if img_size:
        iw, ih = img_size
        base = min(iw, ih) * DEFAULT_FORBIDDEN_RATIO
    else:
        base = 15.0
    r = min(base, MAX_FORBIDDEN_RADIUS)
    r = min(r, min_edge / 4.0)
    return max(r, 0.0)


def _safe_radius(cx, cy, gt_boxes, r_wanted,
                 margin=FORBIDDEN_MARGIN, min_r=MIN_FORBIDDEN_RADIUS):
    if not gt_boxes:
        return r_wanted if r_wanted >= min_r else None
    min_d = _min_dist_to_gt_corners(cx, cy, gt_boxes)
    r_safe = min(r_wanted, min_d - margin)
    if r_safe < min_r:
        return None
    return float(r_safe)


class LearningRules:
    """
    五类规则：
      forbidden_zones:   {img: [{"cx","cy","r","source","created"}]}
      label_corrections: {img: [{"old_xyxy","new_xyxy","new_cls_id",
                                 "new_cls_name","source","created"}]}
      miss_points:       {img: [{"xyxy","cls_id","cls_name","repeat",
                                 "source","created"}]}
      box_adjustments:   {img: [{"kind":"too_big"/"too_small",
                                 "old_xyxy","new_xyxy","source","created"}]}
      convergences:      {img: [{"from_xyxy","to_xyxy","from_cls","to_cls",
                                 "source","created"}]}
    """

    def __init__(self, path):
        self.path = Path(path)
        self.data = {
            "forbidden_zones": {},
            "label_corrections": {},
            "miss_points": {},
            "box_adjustments": {},
            "convergences": {},
        }
        self.load()

    # ==================== 存取 ====================
    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                for k in ("forbidden_zones", "label_corrections", "miss_points",
                          "box_adjustments", "convergences"):
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
                               img_size=None, source="manual"):
        x1, y1, x2, y2 = box_xyxy
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        r_wanted = _compute_wanted_radius(box_xyxy, img_size)
        k = self._key(img_name)
        self.data["forbidden_zones"].setdefault(k, [])
        added = 0; skipped = 0
        for (cx, cy) in corners:
            r_safe = _safe_radius(cx, cy, gt_boxes, r_wanted)
            if r_safe is None:
                skipped += 1; continue
            self.data["forbidden_zones"][k].append({
                "cx": float(cx), "cy": float(cy), "r": r_safe,
                "source": source,
                "created": datetime.now().isoformat(timespec="seconds"),
            })
            added += 1
        self.save()
        return added, skipped

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

    def clear_forbidden(self, img_name=None):
        if img_name is None:
            self.data["forbidden_zones"] = {}
        else:
            self.data["forbidden_zones"].pop(self._key(img_name), None)
        self.save()

    # ==================== 标签修正 ====================
    def add_label_correction(self, img_name, old_xyxy, new_xyxy,
                             new_cls_id, new_cls_name, source="manual"):
        k = self._key(img_name)
        self.data["label_corrections"].setdefault(k, []).append({
            "old_xyxy": [float(v) for v in old_xyxy],
            "new_xyxy": [float(v) for v in new_xyxy],
            "new_cls_id": int(new_cls_id),
            "new_cls_name": str(new_cls_name),
            "source": source,
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        self.save()

    def apply_corrections(self, img_name, boxes, iou_thr=0.5,
                          source_filter=None):
        corrs = self.data["label_corrections"].get(self._key(img_name), [])
        if source_filter is not None:
            corrs = [c for c in corrs if c.get("source") == source_filter]
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

    # ==================== 漏检点 ====================
    def add_miss_point(self, img_name, xyxy, cls_id, cls_name,
                       repeat=DEFAULT_MISS_REPEAT, source="manual"):
        k = self._key(img_name)
        self.data["miss_points"].setdefault(k, []).append({
            "xyxy": [float(v) for v in xyxy],
            "cls_id": int(cls_id),
            "cls_name": str(cls_name),
            "repeat": int(repeat),
            "source": source,
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

    # ==================== 偏大 / 偏小 ====================
    def add_box_adjustment(self, img_name, kind, old_xyxy, new_xyxy,
                           source="manual"):
        """
        kind: "too_big" / "too_small"
        """
        assert kind in ("too_big", "too_small"), kind
        k = self._key(img_name)
        self.data["box_adjustments"].setdefault(k, []).append({
            "kind": kind,
            "old_xyxy": [float(v) for v in old_xyxy],
            "new_xyxy": [float(v) for v in new_xyxy],
            "source": source,
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        self.save()

    def apply_box_adjustments(self, img_name, boxes, iou_thr=0.5,
                              source_filter=None):
        """按 IoU 匹配 old_xyxy，替换成 new_xyxy"""
        adjs = self.data["box_adjustments"].get(self._key(img_name), [])
        if source_filter is not None:
            adjs = [a for a in adjs if a.get("source") == source_filter]
        if not adjs:
            return list(boxes)
        out = []
        for b in boxes:
            matched = None
            for a in adjs:
                if compute_iou(b["xyxy"], a["old_xyxy"]) > iou_thr:
                    matched = a
                    break
            if matched:
                nb = dict(b)
                nb["xyxy"] = list(matched["new_xyxy"])
                out.append(nb)
            else:
                out.append(b)
        return out

    def get_box_adjustments(self, img_name):
        return list(self.data["box_adjustments"].get(self._key(img_name), []))

    def clear_box_adjustments(self, img_name=None):
        if img_name is None:
            self.data["box_adjustments"] = {}
        else:
            self.data["box_adjustments"].pop(self._key(img_name), None)
        self.save()

    # ==================== 误检收敛 ====================
    def add_convergence(self, img_name, from_xyxy, to_xyxy,
                        from_cls, to_cls, source="manual"):
        """
        记 A→B 映射，同时派生：
          - 禁区：A 的 4 顶点
          - 漏检点：B 追加到 miss_points
        """
        k = self._key(img_name)
        self.data["convergences"].setdefault(k, []).append({
            "from_xyxy": [float(v) for v in from_xyxy],
            "to_xyxy": [float(v) for v in to_xyxy],
            "from_cls": str(from_cls),
            "to_cls": str(to_cls),
            "source": source,
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        self.save()

    def get_convergences(self, img_name):
        return list(self.data["convergences"].get(self._key(img_name), []))

    def clear_convergences(self, img_name=None):
        if img_name is None:
            self.data["convergences"] = {}
        else:
            self.data["convergences"].pop(self._key(img_name), None)
        self.save()

    def expand_convergence_to_rules(self, img_name, gt_boxes, img_size=None,
                                    source=None):
        """
        把 convergences 展开成禁区 + 漏检点。
        source=None 时用 convergence 自己的 source。
        """
        convs = self.data["convergences"].get(self._key(img_name), [])
        if source is not None:
            convs = [c for c in convs if c.get("source") == source]
        added_fz = 0; added_mp = 0
        for c in convs:
            a, b = c["from_xyxy"], c["to_xyxy"]
            # 禁区
            a1, added = self.add_forbidden_from_box(
                img_name, a, gt_boxes, img_size=img_size,
                source=c.get("source", "manual"))
            added_fz += added
            # 漏检点
            from core import CLASS_NAME_TO_ID
            cid = CLASS_NAME_TO_ID.get(c["to_cls"], 0)
            self.add_miss_point(img_name, b, cid, c["to_cls"],
                                source=c.get("source", "manual"))
            added_mp += 1
        return added_fz, added_mp

    # ==================== 汇总 ====================
    def has_any(self, img_name=None, source_filter=None):
        if img_name is None:
            return self.count(source_filter) > 0
        k = self._key(img_name)
        for bucket in ("forbidden_zones", "label_corrections", "miss_points",
                       "box_adjustments", "convergences"):
            items = self.data[bucket].get(k, [])
            if source_filter is None:
                if items: return True
            else:
                if any(i.get("source") == source_filter for i in items):
                    return True
        return False

    def _count_bucket(self, name, source_filter=None):
        total = 0
        for items in self.data[name].values():
            if source_filter is None:
                total += len(items)
            else:
                total += sum(1 for i in items if i.get("source") == source_filter)
        return total

    def count_forbidden(self, source_filter=None):
        return self._count_bucket("forbidden_zones", source_filter)

    def count_corrections(self, source_filter=None):
        return self._count_bucket("label_corrections", source_filter)

    def count_miss_points(self, source_filter=None):
        return self._count_bucket("miss_points", source_filter)

    def count_box_adjustments(self, source_filter=None):
        return self._count_bucket("box_adjustments", source_filter)

    def count_convergences(self, source_filter=None):
        return self._count_bucket("convergences", source_filter)

    def count(self, source_filter=None):
        return (self.count_forbidden(source_filter)
                + self.count_corrections(source_filter)
                + self.count_miss_points(source_filter)
                + self.count_box_adjustments(source_filter)
                + self.count_convergences(source_filter))

    def all_names(self):
        keys = set()
        for bucket in ("forbidden_zones", "label_corrections", "miss_points",
                       "box_adjustments", "convergences"):
            keys |= set(self.data[bucket].keys())
        return keys

    def clear_by_source(self, source):
        """清掉所有指定来源的规则（如 'auto'）"""
        for bucket in ("forbidden_zones", "label_corrections", "miss_points",
                       "box_adjustments", "convergences"):
            for k in list(self.data[bucket].keys()):
                self.data[bucket][k] = [
                    i for i in self.data[bucket][k]
                    if i.get("source") != source
                ]
                if not self.data[bucket][k]:
                    del self.data[bucket][k]
        self.save()

    def clear_all(self):
        self.data = {
            "forbidden_zones": {},
            "label_corrections": {},
            "miss_points": {},
            "box_adjustments": {},
            "convergences": {},
        }
        self.save()