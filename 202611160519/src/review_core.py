# review_core.py
# 开卷学习核心：评估 / 微调 / 导出 / 人工错误清单 / 日志 / 进度
# v1.3:
#   - _evaluate_all 每 5 张刷新一次 GT 视图（F1 即时更新）
#   - _is_clean 保留严格口径（零误检/零类别错），仅用于学习闭环的“学会”判定
import json, shutil, time, traceback
from datetime import datetime
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from core import evaluate_one, dedupe_boxes, PROJECT_ROOT


PASS_F1_THR = 0.9


class ReviewCoreMixin:
    PASS_THR = 0.9
    PASS_EPS = 1e-6

    # ==================== 基础工具 ====================
    def _is_pass(self, f1):
        return f1 >= self.PASS_THR - self.PASS_EPS

    def _is_clean(self, m):
        """
        严格通过标准：F1 达标 且 零误检 且 零类别错。
        用于学习闭环判定“学会了”。
        """
        if m is None:
            return False
        if not self._is_pass(m.get("f1", 0)):
            return False
        if m.get("extra_n", 0) > 0:
            return False
        if m.get("wrong_cls", 0) > 0:
            return False
        return True

    @staticmethod
    def _fmt_time(s):
        s = int(s); h = s // 3600; m = (s % 3600) // 60; sec = s % 60
        if h > 0: return f"{h}h{m}m"
        if m > 0: return f"{m}m{sec}s"
        return f"{sec}s"

    def _refresh_all_gt_views(self):
        try:
            self.refresh_gt_list()
        except Exception as e:
            print(f"[刷新自己失败] {e}")
        if hasattr(self.app, "refresh_all_gt_views"):
            try:
                self.app.refresh_all_gt_views()
            except Exception as e:
                print(f"[刷新其他视图失败] {e}")

    def _set_progress(self, pct, text):
        def do():
            try:
                self.progress_var.set(pct)
                self.progress_text.set(text)
            except Exception:
                pass
        try: self.after(0, do)
        except Exception: pass

    def _log_msg(self, msg, tag=None):
        try:
            self.log.config(state="normal")
            prefix = f"[{datetime.now().strftime('%H:%M:%S')}] "
            self.log.insert("end", prefix + msg + "\n", tag or "")
            self.log.see("end")
            self.log.config(state="disabled")
        except Exception:
            pass

    def _log_line(self, msg, tag=None):
        try:
            self.log.config(state="normal")
            self.log.insert("end", msg + "\n", tag or "")
            self.log.see("end")
            self.log.config(state="disabled")
        except Exception:
            pass

    # ==================== 人工错误清单 ====================
    def _load_manual_errors(self):
        if self._manual_errors_path.exists():
            try:
                with open(self._manual_errors_path, "r", encoding="utf-8") as f:
                    self._manual_errors = json.load(f)
            except Exception:
                self._manual_errors = {}

    def _save_manual_errors(self):
        self._manual_errors_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._manual_errors_path, "w", encoding="utf-8") as f:
            json.dump(self._manual_errors, f, ensure_ascii=False, indent=2)

    def _add_manual_error(self, key, pred_cls, pred_conf, wrong_kind="extra"):
        rec = self._manual_errors.get(key, {})
        rec.update({
            "pred_cls": pred_cls,
            "pred_conf": pred_conf,
            "wrong_kind": wrong_kind,
            "marked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cleared": False,
            "cleared_streak": 0,
        })
        self._manual_errors[key] = rec
        self._save_manual_errors()
        self._refresh_intensive_btn()

    def _check_manual_errors(self, result):
        details = []
        for key, info in self._manual_errors.items():
            if info.get("cleared"):
                continue
            m = result.get(key)
            if m is None:
                info["cleared_streak"] = 0
                details.append((key, "未评估(图片丢失?)", 0.0))
                continue
            f1 = m.get("f1", 0)
            if self._is_clean(m):
                info["cleared_streak"] = info.get("cleared_streak", 0) + 1
                if info["cleared_streak"] >= 3:
                    info["cleared"] = True
                    details.append((key, "✅ 已解决(连续3次)", f1))
                else:
                    details.append((key, f"OK {info['cleared_streak']}/3", f1))
            else:
                info["cleared_streak"] = 0
                details.append((key, "❌ 仍错误", f1))
        self._save_manual_errors()
        all_cleared = (not self._manual_errors) or \
                      all(v.get("cleared") for v in self._manual_errors.values())
        return all_cleared, details

    def _refresh_intensive_btn(self):
        n_pending = sum(1 for v in self._manual_errors.values() if not v.get("cleared"))
        try:
            self.btn_intensive.config(text=f"🎯 高强度训练 ({n_pending})")
        except Exception:
            pass

    # ==================== 错误分析 ====================
    def _analyze_errors(self, results):
        total_correct = sum(m.get("correct", 0) for m in results.values())
        total_missed = sum(m.get("missed_n", 0) for m in results.values())
        total_extra = sum(m.get("extra_n", 0) for m in results.values())
        total_cls_err = sum(m.get("wrong_cls", 0) for m in results.values())
        req_violated = sum(1 for m in results.values() if m.get("required_violated"))

        total_gt = total_correct + total_missed + total_cls_err
        total_pred = total_correct + total_extra + total_cls_err
        miss_rate = total_missed / total_gt if total_gt > 0 else 0
        extra_rate = total_extra / total_pred if total_pred > 0 else 0
        cls_err_rate = total_cls_err / total_gt if total_gt > 0 else 0

        reasons = []
        if req_violated > 0:
            reasons.append(f"有 {req_violated} 张必要框漏检/类别错")
            cls_w, box_w, dfl_w = 3.0, 5.0, 1.5
        elif miss_rate > 0.30:
            cls_w, box_w, dfl_w = 2.0, 5.0, 1.5
            reasons.append(f"漏检率 {miss_rate:.1%} 偏高")
        elif extra_rate > 0.30:
            cls_w, box_w, dfl_w = 1.0, 10.0, 2.0
            reasons.append(f"误检率 {extra_rate:.1%} 偏高")
        elif cls_err_rate > 0.20:
            cls_w, box_w, dfl_w = 2.5, 5.0, 1.5
            reasons.append(f"类别错率 {cls_err_rate:.1%} 偏高")
        else:
            cls_w, box_w, dfl_w = 1.5, 7.5, 1.5
            reasons.append("错误分布均衡，使用默认权重")

        return {
            "cls": cls_w, "box": box_w, "dfl": dfl_w,
            "miss_rate": miss_rate, "extra_rate": extra_rate,
            "cls_err_rate": cls_err_rate,
            "total_correct": total_correct, "total_missed": total_missed,
            "total_extra": total_extra, "total_cls_err": total_cls_err,
            "req_violated_count": req_violated,
            "reasons": reasons,
        }

    def _print_analysis(self, analysis):
        self._log_msg("【错误分析】", "title")
        self._log_line(
            f"    命中: {analysis['total_correct']}  "
            f"漏检: {analysis['total_missed']}  "
            f"误检: {analysis['total_extra']}  "
            f"类别错: {analysis['total_cls_err']}", "dim")
        self._log_line(
            f"    漏检率: {analysis['miss_rate']:.1%}  "
            f"误检率: {analysis['extra_rate']:.1%}  "
            f"类别错率: {analysis['cls_err_rate']:.1%}", "dim")
        for r in analysis["reasons"]:
            self._log_line(f"    → {r}", "warn")
        self._log_msg(
            f"  权重: cls={analysis['cls']} box={analysis['box']} dfl={analysis['dfl']}",
            "ok")

    def _log_diagnosis(self, m, prefix="  "):
        correct = m.get("correct", 0); wrong_cls = m.get("wrong_cls", 0)
        missed_n = m.get("missed_n", 0); extra_n = m.get("extra_n", 0)
        total = correct + wrong_cls + missed_n
        f1_val = m.get("f1", 0)
        violated = m.get("required_violated", False)
        warn = "  [必要框漏检，0 分]" if violated else ""
        tag = "err" if violated else "dim"
        self._log_line(
            f"{prefix}逐框: 正确 {correct}/{total}  F1={f1_val:.3f}{warn}", tag)

    # ==================== 评估 ====================
    def _evaluate_all(self, progress_range=(0, 100), save_preds=True):
        gt = self.app.ground_truth
        items = gt.all_reviewed_items()
        result = {}
        total = len(items)
        p_start, p_end = progress_range
        for i, (key, rec) in enumerate(items, 1):
            if self.stop_flag: break
            pct = p_start + (i / max(total, 1)) * (p_end - p_start)
            self._set_progress(pct, f"评估 {i}/{total}")
            img_path = gt.img_dir / rec["file"]
            if not img_path.exists():
                self._log_msg(f"  [{i}/{total}] {key} 图片缺失", "err")
                gt.update_score(key, 0.0, "img_missing")
                continue
            img_size = None
            try:
                with Image.open(img_path) as im:
                    img_size = im.size
            except Exception as e:
                self._log_msg(f"  [{i}/{total}] {key} 读取尺寸失败: {e}", "warn")
            try:
                preds = self.app.detector.predict(str(img_path), conf=0.3)
            except Exception as e:
                self._log_msg(f"  [{i}/{total}] {key} 检测失败: {e}", "err")
                continue
            m = evaluate_one(rec["boxes"], preds, iou_thr=0.5,
                             img_size=img_size, use_area_weight=True,
                             enforce_required=True)
            m["_gt_boxes"] = rec["boxes"]
            if save_preds: m["_preds"] = preds
            result[key] = m

            # ===== 每 5 张刷新一次 GT 视图（F1 即时更新）=====
            if i % 5 == 0:
                try:
                    self.after(0, self._refresh_all_gt_views)
                except Exception:
                    pass
        # 收尾刷新
        try:
            self.after(0, self._refresh_all_gt_views)
        except Exception:
            pass
        return result

    # ==================== 导出 / 微调 ====================
    def _export_all_gt_to_training(self, before_results=None):
        app = self.app
        img_dir = Path(app.auto_img_dir.get())
        lbl_dir = Path(app.auto_labels_dir.get())
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for key, rec in app.ground_truth.all_reviewed_items():
            src = app.ground_truth.img_dir / rec["file"]
            if not src.exists(): continue
            boxes, _ = dedupe_boxes(rec["boxes"])
            required_boxes = [b for b in boxes if b.get("required", False)]
            non_required = [b for b in boxes if not b.get("required", False)]

            use_focus = False
            if required_boxes and non_required and before_results:
                m = before_results.get(key)
                if m is not None:
                    gt_boxes_all = m.get("_gt_boxes", rec["boxes"])
                    matched_ids = set(i for i, j, iou in m.get("tp", []))
                    non_req_ids = set(idx for idx, gb in enumerate(gt_boxes_all)
                                      if not gb.get("required", False))
                    if non_req_ids:
                        hit_rate = len(matched_ids & non_req_ids) / len(non_req_ids)
                        use_focus = hit_rate >= 0.8
                    else:
                        use_focus = True
                else:
                    use_focus = True
            elif required_boxes and not non_required:
                use_focus = True

            export_boxes = required_boxes if (use_focus and required_boxes) else boxes
            new_stem = f"gt_{Path(rec['file']).stem}"
            dst_img = img_dir / (new_stem + Path(rec['file']).suffix)
            dst_lbl = lbl_dir / (new_stem + ".txt")
            try:
                shutil.copy2(src, dst_img)
                iw, ih = Image.open(src).size
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    for b in export_boxes:
                        x1, y1, x2, y2 = b["xyxy"]
                        cx = (x1+x2)/2/iw; cy = (y1+y2)/2/ih
                        w = (x2-x1)/iw; h = (y2-y1)/ih
                        f.write(f"{b['cls_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                exported += 1
            except Exception as e:
                print(f"[导出失败] {key}: {e}")
        return exported

    def _run_finetune(self, epochs, progress_range=(30, 70),
                      loss_weights=None, lr0=0.001):
        app = self.app
        best_pt_obj = Path(app.weights_pred.get())
        if not best_pt_obj.exists():
            self._log_msg(f"  best.pt 不存在", "err"); return False

        ts = datetime.now().strftime("%m%d_%H%M%S")
        p_start, p_end = progress_range
        try:
            timeout_min = int(self.finetune_timeout.get())
        except Exception:
            timeout_min = 10
        timeout_sec = timeout_min * 60

        cls_w = loss_weights.get("cls", 1.5) if loss_weights else 1.5
        box_w = loss_weights.get("box", 7.5) if loss_weights else 7.5
        dfl_w = loss_weights.get("dfl", 1.5) if loss_weights else 1.5

        state = {"start": time.time(), "stop_requested": False, "timed_out": False}
        class _TimeoutStop(Exception): pass

        try:
            self._log_msg(f"  微调 {epochs} epoch | lr0={lr0}", "dim")
            self._set_progress(p_start, "微调准备中...")
            m = YOLO(str(best_pt_obj))

            def on_batch_end(trainer):
                if state["stop_requested"]:
                    raise _TimeoutStop("timeout")
                elapsed = time.time() - state["start"]
                if elapsed > timeout_sec:
                    state["stop_requested"] = True
                    state["timed_out"] = True
                    for attr in ("stop", "stop_training", "should_stop"):
                        try: setattr(trainer, attr, True)
                        except Exception: pass
                    self._log_msg(f"  超时 {timeout_min} 分，强制中断", "warn")
                    raise _TimeoutStop("timeout")
                try:
                    ce = trainer.epoch; te = trainer.epochs
                    try: bpe = len(trainer.train_loader)
                    except Exception: bpe = 1
                    cb = getattr(trainer, "batch_idx", 0)
                    done = ce * bpe + cb + 1
                    total = te * bpe if te else 1
                    pct = p_start + (done / max(total, 1)) * (p_end - p_start)
                    per = elapsed / done if done > 0 else 0
                    remain = per * (total - done)
                    if remain > (timeout_sec - elapsed):
                        remain = timeout_sec - elapsed
                    def ui():
                        self._set_progress(pct,
                            f"微调 Epoch {ce+1}/{te}  已用 {self._fmt_time(elapsed)}  剩余 {self._fmt_time(remain)}")
                    self.after(0, ui)
                except Exception:
                    pass

            m.add_callback("on_train_batch_end", on_batch_end)

            aug_n = len(list(Path(app.auto_img_dir.get()).glob("aug_*")))
            use_mosaic = 0.5 if aug_n > 100 else 1.0

            interrupted = False
            try:
                m.train(data=app.data_yaml.get(), epochs=epochs,
                        imgsz=int(app.imgsz.get()), batch=int(app.batch.get()),
                        device=app.device.get(), project=app.project.get(),
                        name=f"learn_{ts}", exist_ok=True,
                        lr0=lr0, lrf=0.01, cls=cls_w, box=box_w, dfl=dfl_w,
                        degrees=10.0, translate=0.15, scale=0.5, fliplr=0.5,
                        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                        mosaic=use_mosaic, mixup=0.05, patience=15, verbose=False)
            except _TimeoutStop:
                interrupted = True
                self._log_msg("  超时中断训练", "warn")
            except Exception as e:
                if "timeout" in str(e).lower() or state["timed_out"]:
                    interrupted = True
                    self._log_msg("  超时中断训练", "warn")
                else:
                    self._log_msg(f"  微调失败: {e}", "err"); return False

            weights_dir = Path(app.project.get()) / f"learn_{ts}" / "weights"
            source_pt = None
            if (weights_dir / "best.pt").exists():
                source_pt = weights_dir / "best.pt"
            elif (weights_dir / "last.pt").exists():
                source_pt = weights_dir / "last.pt"
            if source_pt is None:
                self._log_msg("  未生成权重", "err"); return False

            ts_bk = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = best_pt_obj.parent / f"{best_pt_obj.stem}_backup_{ts_bk}.pt"
            try:
                shutil.copy2(best_pt_obj, backup)
                self._log_msg(f"  备份: {backup.name}", "dim")
            except Exception as e:
                self._log_msg(f"  备份失败: {e}", "warn")

            shutil.copy2(source_pt, best_pt_obj)
            self._last_backup_path = backup
            self._log_msg(f"  已更新权重: {best_pt_obj.name}", "ok")
            return True
        except Exception:
            self._log_msg(f"  微调异常: {traceback.format_exc()}", "err")
            return False

    def _reload_model(self):
        try:
            from predict import Detector
            self.app.detector = Detector(self.app.weights_pred.get())
            self.after(0, self.app.gt_tab.refresh_model_info)
            self._log_msg("  已重新加载模型", "dim")
            return True
        except Exception as e:
            self._log_msg(f"  重载失败: {e}", "err"); return False

    def _rollback_from_backup(self):
        backup = getattr(self, "_last_backup_path", None)
        if not backup or not Path(backup).exists():
            self._log_msg("  无法回退：没找到备份文件", "err")
            return False
        try:
            shutil.copy2(backup, self.app.weights_pred.get())
            self._log_msg(f"  已回退到: {Path(backup).name}", "warn")
            self._reload_model()
            return True
        except Exception as e:
            self._log_msg(f"  回退失败: {e}", "err")
            return False