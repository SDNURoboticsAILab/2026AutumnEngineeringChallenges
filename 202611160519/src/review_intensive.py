# review_intensive.py
# 高强度训练 + 完美训练
# v1.4:
#   - 训练时应用学习规则（禁区 / 修正标签 / 漏检点）
#   - 漏检图超采样（复制 repeat 份）
#   - 训练期间静默，不画框；训练后才评估
import shutil, threading, traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

from PIL import Image
from ultralytics import YOLO


PASS_F1_THR = 0.9


class ReviewIntensiveMixin:
    # ==================== 高强度训练 ====================
    def intensive_train(self):
        if self.app.detector is None:
            messagebox.showwarning("提示", "请先加载模型"); return
        pending = [k for k, v in self._manual_errors.items() if not v.get("cleared")]
        if not pending:
            messagebox.showinfo("提示",
                "当前没有待解决的人工错误\n\n"
                "① 对比窗口点选预测框 →「❌ 标记选中框为多余」\n"
                "② 查看「📋 人工错误清单」")
            return
        if self.running:
            messagebox.showinfo("提示", "已有任务运行中"); return

        self.running = True
        self.stop_flag = False
        self.btn_once.config(state="disabled")
        self.btn_loop.config(state="disabled")
        self.btn_intensive.config(state="disabled")
        self.btn_perfect.config(state="disabled")
        self.btn_stop.config(state="normal")

        def _worker():
            try:
                self._do_intensive_train()
            except Exception:
                self._log_msg(f"高强度训练异常: {traceback.format_exc()}", "err")
            finally:
                self.running = False; self.stop_flag = False
                self.after(0, lambda: self.btn_once.config(state="normal"))
                self.after(0, lambda: self.btn_loop.config(state="normal"))
                self.after(0, lambda: self.btn_intensive.config(state="normal"))
                self.after(0, lambda: self.btn_perfect.config(state="normal"))
                self.after(0, lambda: self.btn_stop.config(state="disabled"))
                self.after(0, lambda: self._set_progress(0, "就绪"))
                self.after(0, self._refresh_intensive_btn)

        threading.Thread(target=_worker, daemon=True).start()

    def _do_intensive_train(self):
        max_iter = int(self.intensive_max_iter.get())
        epochs = int(self.intensive_epochs.get())
        lr0 = float(self.intensive_lr.get())

        self._log_msg("══════════ 高强度训练开始 ══════════", "title")
        n_pending = sum(1 for v in self._manual_errors.values() if not v.get("cleared"))
        self._log_msg(f"  待解决人工错误: {n_pending} 张", "warn")

        rules = getattr(self.app, "learning_rules", None)
        if rules is not None and rules.count() > 0:
            self._log_msg(
                f"  学习规则: 禁区 {rules.count_forbidden()}  |  "
                f"修正 {rules.count_corrections()}  |  "
                f"漏检 {rules.count_miss_points()}", "dim")

        self._log_msg(
            f"  最多 {max_iter} 轮 | 每轮 {epochs} epoch | lr0={lr0} | "
            f"退出: F1≥0.9 且零误检/零类别错，连续 3 次不复发", "dim")

        for it in range(max_iter):
            if self.stop_flag:
                self._log_msg("  收到停止请求", "warn"); break
            pending_keys = [k for k, v in self._manual_errors.items() if not v.get("cleared")]
            if not pending_keys:
                self._log_msg("  ✅ 所有人工错误已解决", "ok"); break

            self._log_line("")
            self._log_msg(f"──── 第 {it+1}/{max_iter} 轮 | 待解决 {len(pending_keys)} 张 ────", "title")
            for k in pending_keys:
                info = self._manual_errors[k]
                self._log_line(
                    f"    · {k}  错误类别={info.get('pred_cls')}  "
                    f"连续OK={info.get('cleared_streak', 0)}/3", "dim")

            exported = self._export_intensive_set(pending_keys)
            self._log_msg(f"  导出到强化训练集: {exported} 张", "dim")
            if exported == 0:
                self._log_msg("  无可训练图，终止", "err"); break

            if not self._run_intensive_finetune(epochs, lr0=lr0): break
            if not self._reload_model(): break
            if self.stop_flag: break

            self._log_msg("  [评估] 重新考核所有已审核 GT...", "title")
            result = self._evaluate_all(progress_range=(0, 100), save_preds=True)

            gt = self.app.ground_truth
            for key, m in result.items():
                st = "pass" if m.get("f1", 0) >= PASS_F1_THR - 1e-6 else "fail"
                gt.update_score(key, m["f1"], st)
            self.after(0, self._refresh_all_gt_views)

            all_cleared, details = self._check_manual_errors(result)
            self._log_line("")
            self._log_msg("  [本轮人工错误检查]", "title")
            for key, msg, f1 in details:
                tag = "ok" if "已解决" in msg else ("dim" if "OK" in msg else "err")
                self._log_line(f"    {key}  F1={f1:.3f}  {msg}", tag)
            self.after(0, self._refresh_intensive_btn)

            if all_cleared:
                self._log_msg("  ✅ 全部人工错误连续 3 次不复发，训练完成！", "ok")
                self.after(0, lambda: messagebox.showinfo(
                    "完成", "所有人工错误都已解决！\n\n（F1≥0.9 且连续 3 次不复发）"))
                break
        else:
            self._log_msg(f"  ⚠ 达到最大迭代 {max_iter} 仍未全部解决", "warn")

        self._log_msg("══════════ 高强度训练结束 ══════════", "title")

    # ==================== 导出强化集 ====================
    def _export_intensive_set(self, keys):
        """
        1. 导出所有已审核 GT（保证不遗忘）
        2. 应用学习规则：标签修正 → 禁区过滤 → 漏检点追加
        3. 漏检图超采样 repeat 份
        4. 重点图（人工错误 + 人工合并 + 引导 + 规则）复制 2~3 份
        5. 引导数据（guidance）额外 3 份
        """
        gt = self.app.ground_truth
        rules = getattr(self.app, "learning_rules", None)
        img_dir = self._intensive_dir / "images"
        lbl_dir = self._intensive_dir / "labels"
        for d in (img_dir, lbl_dir):
            if d.exists():
                for p in d.iterdir():
                    try: p.unlink()
                    except Exception: pass
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        all_items = gt.all_reviewed_items()
        base_count = 0
        miss_extra = 0

        for key, rec in all_items:
            src = gt.img_dir / rec["file"]
            if not src.exists(): continue
            try:
                iw, ih = Image.open(src).size
            except Exception:
                continue

            boxes = self._build_boxes_with_rules(key, rec, rules)

            # 基础一份
            try:
                shutil.copy2(src, img_dir / rec["file"])
                self._write_label(
                    lbl_dir / (Path(rec["file"]).stem + ".txt"),
                    boxes, iw, ih)
                base_count += 1
            except Exception as e:
                print(f"[导出基础集失败] {key}: {e}")

            # 漏检图超采样
            if rules is not None:
                mps = rules.get_miss_points(key)
                if mps:
                    repeat = max(int(mp.get("repeat", 8)) for mp in mps)
                    stem = Path(rec["file"]).stem
                    suffix = Path(rec["file"]).suffix
                    for i in range(repeat):
                        dst_img = img_dir / f"{stem}_miss{i}{suffix}"
                        dst_lbl = lbl_dir / f"{stem}_miss{i}.txt"
                        try:
                            shutil.copy2(src, dst_img)
                            self._write_label(dst_lbl, boxes, iw, ih)
                            miss_extra += 1
                        except Exception as e:
                            print(f"[漏检超采样失败] {key}: {e}")

        # ===== 重点图（人工错误 + 人工合并 + 规则） =====
        focus_keys = set(keys)
        merge_log = getattr(self.app, "merge_log", None)
        if merge_log is not None:
            for k in merge_log.all_names():
                focus_keys.add(k)
        if rules is not None:
            for k in rules.all_names():
                focus_keys.add(k)

        focus_count = 0
        for key in focus_keys:
            rec = gt.get_by_key(key)
            if not rec: continue
            src = gt.img_dir / rec["file"]
            if not src.exists(): continue
            try:
                iw, ih = Image.open(src).size
            except Exception:
                continue
            dup_n = 3 if (merge_log and merge_log.is_merged(key)) else 2
            for dup in range(dup_n):
                stem = Path(rec["file"]).stem
                suffix = Path(rec["file"]).suffix
                dst_img = img_dir / f"{stem}_focus{dup}{suffix}"
                dst_lbl = lbl_dir / f"{stem}_focus{dup}.txt"
                try:
                    shutil.copy2(src, dst_img)
                    boxes = self._build_boxes_with_rules(key, rec, rules)
                    self._write_label(dst_lbl, boxes, iw, ih)
                    focus_count += 1
                except Exception as e:
                    print(f"[导出强化副本失败] {key}: {e}")

        # ===== 引导数据（guidance） =====
        guidance_count = 0
        guidance = getattr(self.app, "guidance", None)
        if guidance is not None and guidance.count() > 0:
            for name in guidance.all_names():
                rec = gt.get_by_key(name)
                if not rec: continue
                src = gt.img_dir / rec["file"]
                if not src.exists(): continue
                try:
                    iw, ih = Image.open(src).size
                except Exception:
                    continue
                g = guidance.get(name)
                boxes = self._build_boxes_with_rules(name, rec, rules)

                from core import compute_iou, CLASS_NAME_TO_ID
                for wc in g.get("wrong_cls", []):
                    for b in boxes:
                        if compute_iou(b["xyxy"], wc["pred_xyxy"]) > 0.5:
                            b["cls_id"] = int(CLASS_NAME_TO_ID.get(
                                wc["to_cls"], b["cls_id"]))
                for dup in g.get("duplicate", []):
                    group = dup.get("group", [])
                    merged = dup.get("merged_xyxy")
                    if not group or not merged: continue
                    boxes = [b for b in boxes
                             if not any(compute_iou(b["xyxy"], gb) > 0.5
                                        for gb in group)]
                    boxes.append({
                        "xyxy": [float(v) for v in merged],
                        "cls_id": int(dup.get("cls_id", 0)),
                        "cls_name": dup.get("cls_name", "obstacle")})
                for ms in g.get("missed", []):
                    boxes.append({
                        "xyxy": [float(v) for v in ms["xyxy"]],
                        "cls_id": int(ms.get("cls_id", 0)),
                        "cls_name": ms.get("cls_name", "obstacle")})

                for dup in range(3):
                    stem = Path(rec["file"]).stem
                    suffix = Path(rec["file"]).suffix
                    dst_img = img_dir / f"{stem}_guide{dup}{suffix}"
                    dst_lbl = lbl_dir / f"{stem}_guide{dup}.txt"
                    try:
                        shutil.copy2(src, dst_img)
                        self._write_label(dst_lbl, boxes, iw, ih)
                        guidance_count += 1
                    except Exception as e:
                        print(f"[导出引导副本失败] {name}: {e}")

        self._log_msg(
            f"  基础集 {base_count} 张  |  重点图 {focus_count} 张  |  "
            f"漏检超采样 {miss_extra} 张  |  引导图 {guidance_count} 张",
            "dim")
        return base_count + focus_count + miss_extra + guidance_count

    @staticmethod
    def _build_boxes_with_rules(key, rec, rules):
        """应用学习规则：修正 → 过滤禁区 → 追加漏检点"""
        boxes = [{"xyxy": list(b["xyxy"]),
                   "cls_id": int(b["cls_id"]),
                   "cls_name": b["cls_name"]}
                 for b in rec["boxes"]]
        if rules is None:
            return boxes
        # 1. 修正标签
        boxes = rules.apply_corrections(key, boxes)
        # 2. 过滤禁区（训练集里本就不该有错误框）
        boxes = rules.filter_boxes_by_forbidden(
            key, boxes, gt_boxes=None, iou_thr=0.5, cls_aware=True)
        # 3. 追加漏检点
        for mp in rules.get_miss_points(key):
            boxes.append({
                "xyxy": list(mp["xyxy"]),
                "cls_id": int(mp["cls_id"]),
                "cls_name": mp["cls_name"],
            })
        return boxes

    @staticmethod
    def _write_label(lbl_path, boxes, iw, ih):
        with open(lbl_path, "w", encoding="utf-8") as f:
            for b in boxes:
                x1, y1, x2, y2 = b["xyxy"]
                cx = (x1 + x2) / 2 / iw
                cy = (y1 + y2) / 2 / ih
                w = (x2 - x1) / iw
                h = (y2 - y1) / ih
                f.write(f"{b['cls_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def _run_intensive_finetune(self, epochs, lr0=0.005):
        app = self.app
        best_pt_obj = Path(app.weights_pred.get())
        if not best_pt_obj.exists():
            self._log_msg(f"  best.pt 不存在", "err"); return False

        tmp_yaml = self._intensive_dir / "data.yaml"
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            f.write(f"path: {self._intensive_dir.as_posix()}\n")
            f.write("train: images\nval: images\nnc: 3\n")
            f.write("names: ['obstacle', 'cola', 'football']\n")

        ts = datetime.now().strftime("%m%d_%H%M%S")
        try:
            self._log_msg(f"  [训练] {epochs} epoch | lr0={lr0} | 强增强 | 静默", "dim")
            self._set_progress(20, "高强度训练准备中...")
            m = YOLO(str(best_pt_obj))

            # 静默：只更新进度条，不画任何框
            def on_batch_end(trainer):
                try:
                    if self.stop_flag:
                        for attr in ("stop", "stop_training", "should_stop"):
                            try: setattr(trainer, attr, True)
                            except Exception: pass
                    ce = trainer.epoch; te = trainer.epochs
                    try: bpe = len(trainer.train_loader)
                    except Exception: bpe = 1
                    cb = getattr(trainer, "batch_idx", 0)
                    done = ce * bpe + cb + 1
                    total = te * bpe if te else 1
                    pct = 20 + (done / max(total, 1)) * 60
                    def ui():
                        self._set_progress(pct, f"高强度训练 Epoch {ce+1}/{te}")
                    self.after(0, ui)
                except Exception:
                    pass

            m.add_callback("on_train_batch_end", on_batch_end)

            m.train(
                data=str(tmp_yaml), epochs=epochs,
                imgsz=int(app.imgsz.get()), batch=int(app.batch.get()),
                device=app.device.get(),
                project=str(self._intensive_dir / "runs"),
                name=f"int_{ts}", exist_ok=True,
                lr0=lr0, lrf=0.05, cls=2.5, box=7.0, dfl=1.5,
                degrees=0.0, translate=0.0, scale=0.0,
                fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                mosaic=0.0, mixup=0.0, copy_paste=0.0,
                patience=50, verbose=False)

            weights_dir = self._intensive_dir / "runs" / f"int_{ts}" / "weights"
            source_pt = None
            if (weights_dir / "best.pt").exists():
                source_pt = weights_dir / "best.pt"
            elif (weights_dir / "last.pt").exists():
                source_pt = weights_dir / "last.pt"
            if source_pt is None:
                self._log_msg("  未生成权重", "err"); return False

            ts_bk = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = best_pt_obj.parent / f"{best_pt_obj.stem}_intbk_{ts_bk}.pt"
            try:
                shutil.copy2(best_pt_obj, backup)
                self._log_msg(f"  备份: {backup.name}", "dim")
            except Exception as e:
                self._log_msg(f"  备份失败: {e}", "warn")

            shutil.copy2(source_pt, best_pt_obj)
            self._last_backup_path = backup
            self._log_msg(f"  已更新权重: {best_pt_obj.name}", "ok")
            return True
        except Exception as e:
            self._log_msg(f"  高强度训练异常: {e}", "err")
            return False

    # ==================== 完美训练 ====================
    def perfect_train(self):
        if self.app.detector is None:
            messagebox.showwarning("提示", "请先加载模型"); return
        reviewed = self.app.ground_truth.all_reviewed_items()
        if not reviewed:
            messagebox.showinfo("提示", "没有已审核的 GT"); return
        if self.running:
            messagebox.showinfo("提示", "已有任务运行中"); return

        if not messagebox.askyesno(
            "⚠ 完美训练（耗时会显著增加）",
            "完美训练目标：\n"
            "  · 漏检率 < 1%\n"
            "  · 误检率 = 0%\n"
            "  · 类别错率 = 0%\n\n"
            "机制：\n"
            "  · 每轮用全部已审核 GT 训练（含学习规则）\n"
            "  · 训练后全量评估，未达标就继续\n"
            "  · 直到三率全部满足或达到最大轮数\n\n"
            "⚠ 这会显著增加训练时间（可能几十分钟到数小时）\n"
            "⚠ 建议先备份 best.pt\n\n"
            "是否继续？"):
            return

        self.running = True
        self.stop_flag = False
        self.btn_once.config(state="disabled")
        self.btn_loop.config(state="disabled")
        self.btn_intensive.config(state="disabled")
        self.btn_perfect.config(state="disabled")
        self.btn_stop.config(state="normal")

        def _worker():
            try:
                self._do_perfect_train()
            except Exception:
                self._log_msg(f"完美训练异常: {traceback.format_exc()}", "err")
            finally:
                self.running = False; self.stop_flag = False
                self.after(0, lambda: self.btn_once.config(state="normal"))
                self.after(0, lambda: self.btn_loop.config(state="normal"))
                self.after(0, lambda: self.btn_intensive.config(state="normal"))
                self.after(0, lambda: self.btn_perfect.config(state="normal"))
                self.after(0, lambda: self.btn_stop.config(state="disabled"))
                self.after(0, lambda: self._set_progress(0, "就绪"))

        threading.Thread(target=_worker, daemon=True).start()

    def _compute_global_metrics(self, result):
        total_correct = sum(m.get("correct", 0) for m in result.values())
        total_missed = sum(m.get("missed_n", 0) for m in result.values())
        total_extra = sum(m.get("extra_n", 0) for m in result.values())
        total_cls_err = sum(m.get("wrong_cls", 0) for m in result.values())
        total_gt = total_correct + total_missed + total_cls_err
        total_pred = total_correct + total_extra + total_cls_err
        return {
            "miss_rate": total_missed / total_gt if total_gt > 0 else 0,
            "extra_rate": total_extra / total_pred if total_pred > 0 else 0,
            "cls_err_rate": total_cls_err / total_gt if total_gt > 0 else 0,
            "total_correct": total_correct,
            "total_missed": total_missed,
            "total_extra": total_extra,
            "total_cls_err": total_cls_err,
        }

    def _do_perfect_train(self):
        max_iter = int(self.perfect_max_iter.get())
        epochs = int(self.perfect_epochs.get())
        lr0 = float(self.perfect_lr.get())
        miss_thr = float(self.perfect_miss_thr.get())

        self._log_msg("══════════ 完美训练开始 ══════════", "title")
        self._log_msg(f"  目标: 漏检率<{miss_thr:.0%}  误检率=0%  类别错率=0%", "warn")
        self._log_msg(f"  最多 {max_iter} 轮 | 每轮 {epochs} epoch | lr0={lr0}", "dim")

        rules = getattr(self.app, "learning_rules", None)
        if rules is not None and rules.count() > 0:
            self._log_msg(
                f"  学习规则: 禁区 {rules.count_forbidden()}  |  "
                f"修正 {rules.count_corrections()}  |  "
                f"漏检 {rules.count_miss_points()}", "dim")

        for it in range(max_iter):
            if self.stop_flag:
                self._log_msg("  收到停止请求", "warn"); break

            self._log_line("")
            self._log_msg(f"──── 第 {it+1}/{max_iter} 轮 ────", "title")

            exported = self._export_intensive_set(keys=[])
            self._log_msg(f"  导出 {exported} 张到训练集", "dim")
            if exported == 0:
                self._log_msg("  无可训练图，终止", "err"); break

            if not self._run_perfect_finetune(epochs, lr0=lr0): break
            if not self._reload_model(): break
            if self.stop_flag: break

            self._log_msg("  [评估] 全量考核...", "title")
            result = self._evaluate_all(progress_range=(0, 100), save_preds=True)

            gt = self.app.ground_truth
            for key, m in result.items():
                st = "pass" if m.get("f1", 0) >= PASS_F1_THR - 1e-6 else "fail"
                gt.update_score(key, m["f1"], st)
            self.after(0, self._refresh_all_gt_views)

            metrics = self._compute_global_metrics(result)
            self._log_msg(
                f"  本轮: 漏检率={metrics['miss_rate']:.2%}  "
                f"误检率={metrics['extra_rate']:.2%}  "
                f"类别错率={metrics['cls_err_rate']:.2%}", "title")
            self._log_line(
                f"    命中 {metrics['total_correct']}  "
                f"漏检 {metrics['total_missed']}  "
                f"误检 {metrics['total_extra']}  "
                f"类别错 {metrics['total_cls_err']}", "dim")

            ok_miss = metrics["miss_rate"] < miss_thr
            ok_extra = metrics["extra_rate"] == 0
            ok_cls = metrics["cls_err_rate"] == 0
            if ok_miss and ok_extra and ok_cls:
                self._log_msg("  🎉 完美达成！", "ok")
                self.after(0, lambda m=metrics: messagebox.showinfo(
                    "完美达成",
                    f"漏检率: {m['miss_rate']:.2%}\n"
                    f"误检率: {m['extra_rate']:.2%}\n"
                    f"类别错率: {m['cls_err_rate']:.2%}\n\n完美训练完成！"))
                break

            reasons = []
            if not ok_miss: reasons.append(f"漏检率 {metrics['miss_rate']:.2%}")
            if not ok_extra: reasons.append(f"误检率 {metrics['extra_rate']:.2%}")
            if not ok_cls: reasons.append(f"类别错率 {metrics['cls_err_rate']:.2%}")
            self._log_msg(f"  ❌ 未达标: {', '.join(reasons)}", "warn")
        else:
            self._log_msg(f"  ⚠ 达到最大迭代 {max_iter}，仍未完美达标", "warn")

        self._log_msg("══════════ 完美训练结束 ══════════", "title")

    def _run_perfect_finetune(self, epochs, lr0=0.003):
        app = self.app
        best_pt_obj = Path(app.weights_pred.get())
        if not best_pt_obj.exists():
            self._log_msg(f"  best.pt 不存在", "err"); return False

        # 完美训练使用强化集目录
        tmp_yaml = self._intensive_dir / "data.yaml"
        if not tmp_yaml.exists():
            with open(tmp_yaml, "w", encoding="utf-8") as f:
                f.write(f"path: {self._intensive_dir.as_posix()}\n")
                f.write("train: images\nval: images\nnc: 3\n")
                f.write("names: ['obstacle', 'cola', 'football']\n")

        ts = datetime.now().strftime("%m%d_%H%M%S")
        try:
            self._log_msg(f"  [训练] {epochs} epoch | lr0={lr0} | 强增强 | 静默", "dim")
            self._set_progress(20, "完美训练准备中...")
            m = YOLO(str(best_pt_obj))

            def on_batch_end(trainer):
                try:
                    if self.stop_flag:
                        for attr in ("stop", "stop_training", "should_stop"):
                            try: setattr(trainer, attr, True)
                            except Exception: pass
                    ce = trainer.epoch; te = trainer.epochs
                    try: bpe = len(trainer.train_loader)
                    except Exception: bpe = 1
                    cb = getattr(trainer, "batch_idx", 0)
                    done = ce * bpe + cb + 1
                    total = te * bpe if te else 1
                    pct = 20 + (done / max(total, 1)) * 60
                    def ui():
                        self._set_progress(pct, f"完美训练 Epoch {ce+1}/{te}")
                    self.after(0, ui)
                except Exception:
                    pass

            m.add_callback("on_train_batch_end", on_batch_end)

            m.train(
                data=str(tmp_yaml), epochs=epochs,
                imgsz=int(app.imgsz.get()), batch=int(app.batch.get()),
                device=app.device.get(),
                project=str(self._intensive_dir / "runs"),
                name=f"perfect_{ts}", exist_ok=True,
                lr0=lr0, lrf=0.05, cls=2.5, box=7.0, dfl=1.5,
                degrees=15.0, translate=0.2, scale=0.6,
                fliplr=0.5, hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
                mosaic=1.0, mixup=0.1, copy_paste=0.1,
                patience=20, verbose=False)

            weights_dir = self._intensive_dir / "runs" / f"perfect_{ts}" / "weights"
            source_pt = None
            if (weights_dir / "best.pt").exists():
                source_pt = weights_dir / "best.pt"
            elif (weights_dir / "last.pt").exists():
                source_pt = weights_dir / "last.pt"
            if source_pt is None:
                self._log_msg("  未生成权重", "err"); return False

            ts_bk = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = best_pt_obj.parent / f"{best_pt_obj.stem}_pfbk_{ts_bk}.pt"
            try:
                shutil.copy2(best_pt_obj, backup)
                self._log_msg(f"  备份: {backup.name}", "dim")
            except Exception as e:
                self._log_msg(f"  备份失败: {e}", "warn")

            shutil.copy2(source_pt, best_pt_obj)
            self._last_backup_path = backup
            self._log_msg(f"  已更新权重: {best_pt_obj.name}", "ok")
            return True
        except Exception as e:
            self._log_msg(f"  完美训练异常: {e}", "err")
            return False