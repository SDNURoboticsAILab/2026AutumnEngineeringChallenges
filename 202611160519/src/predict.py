# predict.py
# 推理器 Detector 类
# v1.3:
#   - apply_guidance：人工引导硬覆盖
#   - auto_resolve：自动合并同类重复框
#   - learning_rules + gt_provider：禁区过滤（豁免匹配 GT 的预测）
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from core import (CLASS_NAMES, CLASS_NAME_TO_ID, compute_iou,
                  auto_resolve_duplicates)


class Detector:
    def __init__(self, weights):
        if not Path(weights).exists():
            raise FileNotFoundError(f"权重文件不存在: {weights}")
        self.model = YOLO(weights)
        self.names = self.model.names

    # ==================== 主入口 ====================
    def predict(self, image, conf=0.25, iou=0.5,
                auto_resolve=False,
                apply_guidance=False, guidance=None, img_name=None,
                learning_rules=None, gt_provider=None):
        try:
            results = self.model.predict(
                image, conf=conf, iou=iou, verbose=False)
        except Exception:
            print("[predict] YOLO 推理失败:")
            print(traceback.format_exc())
            return []

        dets = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                dets.append({
                    "xyxy": box.xyxy[0].cpu().numpy().tolist(),
                    "conf": float(box.conf),
                    "cls_id": cls_id,
                    "cls_name": r.names[cls_id],
                })

        # 1. 自动合并同类重复框
        if auto_resolve and dets:
            try:
                dets, _actions = auto_resolve_duplicates(dets)
            except Exception as e:
                print(f"[predict] auto_resolve 失败: {e}")

        # 2. 人工引导硬覆盖
        if apply_guidance and guidance and img_name and guidance.has(img_name):
            try:
                dets = self._apply_guidance(dets, guidance.get(img_name))
            except Exception as e:
                print(f"[predict] 应用引导失败: {e}")

        # 3. 禁区过滤（豁免匹配上 GT 的预测）
        if learning_rules is not None and img_name and dets:
            try:
                gt_boxes_for_filter = None
                if gt_provider is not None:
                    try:
                        gt_boxes_for_filter = gt_provider(img_name)
                    except Exception as e:
                        print(f"[predict] gt_provider 失败: {e}")
                dets = learning_rules.filter_boxes_by_forbidden(
                    img_name, dets,
                    gt_boxes=gt_boxes_for_filter,
                    iou_thr=0.5, cls_aware=True)
            except Exception as e:
                print(f"[predict] 禁区过滤失败: {e}")

        return dets

    # ==================== 人工引导覆盖 ====================
    @staticmethod
    def _apply_guidance(dets, rec):
        dets = [dict(d) for d in dets]

        for wc in rec.get("wrong_cls", []):
            for d in dets:
                if compute_iou(d["xyxy"], wc["pred_xyxy"]) > 0.5:
                    to_cls = wc.get("to_cls", d["cls_name"])
                    d["cls_id"] = CLASS_NAME_TO_ID.get(to_cls, d["cls_id"])
                    d["cls_name"] = to_cls

        for dup in rec.get("duplicate", []):
            group = dup.get("group", [])
            merged = dup.get("merged_xyxy")
            if not group or not merged:
                continue
            new_dets = []
            for d in dets:
                if any(compute_iou(d["xyxy"], g) > 0.5 for g in group):
                    continue
                new_dets.append(d)
            dets = new_dets
            cls_id = int(dup.get("cls_id", 0))
            dets.append({
                "xyxy": merged,
                "conf": 1.0,
                "cls_id": cls_id,
                "cls_name": CLASS_NAMES.get(cls_id, "obstacle"),
            })

        for ms in rec.get("missed", []):
            xyxy = ms.get("xyxy")
            if not xyxy:
                continue
            cls_id = int(ms.get("cls_id", 0))
            if any(compute_iou(xyxy, d["xyxy"]) > 0.8 for d in dets):
                continue
            dets.append({
                "xyxy": [float(v) for v in xyxy],
                "conf": 1.0,
                "cls_id": cls_id,
                "cls_name": CLASS_NAMES.get(cls_id, "obstacle"),
            })

        return dets

    # ==================== 保存图片 ====================
    def predict_and_save(self, source, output_dir, conf=0.25):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.predict(
            source=source, conf=conf, save=True,
            project=str(Path(output_dir).parent),
            name=Path(output_dir).name,
            exist_ok=True, verbose=False)
        return output_dir


# ========== 命令行入口 ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO 推理脚本")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--source", required=True, help="图片/目录路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--out", default="results/predict", help="输出目录")
    args = parser.parse_args()

    det = Detector(args.weights)
    det.predict_and_save(args.source, args.out, conf=args.conf)
    print(f"检测完成，结果保存在: {args.out}")