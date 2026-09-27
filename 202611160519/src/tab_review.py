# tab_review.py
# 开卷学习主 Tab（UI + GT 列表 + 只考试/闭环）
# v1.5:
#   - 智能杂交按钮 → 打开模型浏览器
#   - 加 💾 保存当前模型 / 📁 另存为
#   - 表格状态只看 F1
#   - ESC 关闭所有子窗口
#   - 去掉旧的 soup_models_dialog / _do_smart_soup / _on_smart_soup_done / _collect_candidate_models
import shutil
import threading, traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from core import PROJECT_ROOT, evaluate_one

from review_core import ReviewCoreMixin
from review_dialogs import ReviewDialogsMixin
from review_intensive import ReviewIntensiveMixin


STATUS_TEXT_SHORT = {
    "pass": "✓通过", "fail": "✗未通过",
    "ok": "✓正常", "miss": "✗漏检", "conflict": "!冲突",
    "over_count": "⚠数量", "unknown": "?", "untested": "-",
    "img_missing": "图片缺失", "required_fail": "必要框漏检",
}

PASS_F1_THR = 0.9


class ReviewTab(ReviewIntensiveMixin, ReviewDialogsMixin, ReviewCoreMixin, ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.max_iter = tk.IntVar(value=5)
        self.epochs_per_iter = tk.IntVar(value=3)
        self.finetune_timeout = tk.IntVar(value=10)
        self.lr_decay = tk.DoubleVar(value=0.5)
        self.auto_rollback = tk.BooleanVar(value=True)
        self.running = False
        self.stop_flag = False
        self.stat_var = tk.StringVar(value="标准答案: 0  |  已审核: 0  |  通过: 0  |  未通过: 0")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text = tk.StringVar(value="就绪")

        self._last_report = None
        self._report_win = None
        self._last_backup_path = None

        # 高强度训练
        self.intensive_max_iter = tk.IntVar(value=20)
        self.intensive_epochs = tk.IntVar(value=5)
        self.intensive_lr = tk.DoubleVar(value=0.005)
        self._intensive_dir = PROJECT_ROOT / "results" / "intensive"
        self._intensive_dir.mkdir(parents=True, exist_ok=True)
        self._manual_errors_path = PROJECT_ROOT / "results" / "manual_errors.json"
        self._manual_errors = {}
        self._load_manual_errors()

        # 完美训练
        self.perfect_max_iter = tk.IntVar(value=10)
        self.perfect_epochs = tk.IntVar(value=10)
        self.perfect_lr = tk.DoubleVar(value=0.003)
        self.perfect_miss_thr = tk.DoubleVar(value=0.01)

        self._build_ui()

    # ==================== 工具 ====================
    def _bind_esc_close(self, win):
        try:
            win.bind("<Escape>", lambda e: self._safe_close(win))
        except Exception:
            pass

    def _safe_close(self, win):
        try:
            if win is not None and win.winfo_exists():
                win.destroy()
        except Exception:
            pass

    # ==================== UI ====================
    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(top, text="IoU:").pack(side=tk.LEFT)
        ttk.Label(top, text="0.5", foreground="#007acc").pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(top, text="合格线 F1:").pack(side=tk.LEFT)
        ttk.Label(top, text="0.9 且零误检/零类别错", foreground="#007acc").pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(top, text="最大迭代:").pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=1, to=20, textvariable=self.max_iter, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="每轮 epoch:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(top, from_=1, to=20, textvariable=self.epochs_per_iter, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="  超时(分):").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(top, from_=1, to=120, textvariable=self.finetune_timeout, width=5).pack(side=tk.LEFT, padx=2)

        row_lr = ttk.Frame(self); row_lr.pack(fill=tk.X, padx=10, pady=(0, 4))
        ttk.Label(row_lr, text="学习率衰减系数:").pack(side=tk.LEFT)
        ttk.Spinbox(row_lr, from_=0.1, to=1.0, increment=0.1,
                    textvariable=self.lr_decay, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row_lr, text="学后变差自动回退",
                        variable=self.auto_rollback).pack(side=tk.LEFT, padx=12)

        # ===== 主按钮栏 =====
        btns = ttk.Frame(self); btns.pack(fill=tk.X, padx=10, pady=4)
        self.btn_once = ttk.Button(btns, text="只考试一次", command=self.review_once)
        self.btn_once.pack(side=tk.LEFT, padx=3)
        self.btn_loop = ttk.Button(btns, text="开卷学习闭环", command=self.review_loop)
        self.btn_loop.pack(side=tk.LEFT, padx=3)
        self.btn_stop = ttk.Button(btns, text="停止", command=self.stop_loop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=3)
        self.btn_report = ttk.Button(btns, text="学习报告", command=self.open_learning_report)
        self.btn_report.pack(side=tk.LEFT, padx=3)

        self.btn_intensive = ttk.Button(btns, text="🎯 高强度训练 (0)",
                                        command=self.intensive_train)
        self.btn_intensive.pack(side=tk.LEFT, padx=3)
        self.btn_manual_err = ttk.Button(btns, text="📋 人工错误清单",
                                         command=self.show_manual_errors)
        self.btn_manual_err.pack(side=tk.LEFT, padx=3)

        self.btn_perfect = ttk.Button(btns, text="💎 完美训练",
                                      command=self.perfect_train)
        self.btn_perfect.pack(side=tk.LEFT, padx=3)

        self.btn_soup = ttk.Button(btns, text="🧬 智能杂交(模型浏览器)",
                                   command=self.open_model_browser)
        self.btn_soup.pack(side=tk.LEFT, padx=3)

        # ===== 新增：保存 / 另存为 =====
        self.btn_save_model = ttk.Button(btns, text="💾 保存当前模型",
                                         command=self.save_current_model)
        self.btn_save_model.pack(side=tk.LEFT, padx=3)
        self.btn_save_as = ttk.Button(btns, text="📁 另存为",
                                      command=self.save_current_model_as)
        self.btn_save_as.pack(side=tk.LEFT, padx=3)

        ttk.Label(btns, textvariable=self.stat_var,
                  font=('Consolas', 10), foreground="#007acc").pack(side=tk.RIGHT, padx=8)

        # ===== 第二行参数 =====
        row2 = ttk.Frame(self); row2.pack(fill=tk.X, padx=10, pady=(0, 4))
        ttk.Label(row2, text="🎯 高强度训练:", foreground="#cc6600").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(row2, text="轮数").pack(side=tk.LEFT)
        ttk.Spinbox(row2, from_=1, to=100, textvariable=self.intensive_max_iter, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="ep").pack(side=tk.LEFT)
        ttk.Spinbox(row2, from_=1, to=100, textvariable=self.intensive_epochs, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="lr0").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.intensive_lr, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(row2, text="  💎 完美训练:", foreground="#cc0066").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Label(row2, text="轮数").pack(side=tk.LEFT)
        ttk.Spinbox(row2, from_=1, to=100, textvariable=self.perfect_max_iter, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="ep").pack(side=tk.LEFT)
        ttk.Spinbox(row2, from_=1, to=100, textvariable=self.perfect_epochs, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="lr0").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.perfect_lr, width=6).pack(side=tk.LEFT, padx=2)

        # 进度条
        prog_frame = ttk.Frame(self); prog_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100,
                        length=400).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(prog_frame, textvariable=self.progress_text,
                  font=('Consolas', 10), foreground="#cc6600").pack(side=tk.LEFT)

        # 主区
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        left = ttk.Frame(main)
        listf = ttk.LabelFrame(left, text="标准答案库（双击查看对比）")
        listf.pack(fill=tk.BOTH, expand=True)
        filter_row = ttk.Frame(listf); filter_row.pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Label(filter_row, text="筛选:").pack(side=tk.LEFT)
        self.gt_filter_var = tk.StringVar(value="全部")
        ttk.Combobox(filter_row, textvariable=self.gt_filter_var,
                     values=["全部", "已审核", "待审核", "未通过", "已通过"],
                     width=10, state="readonly").pack(side=tk.LEFT, padx=4)
        self.gt_filter_var.trace_add('write', lambda *a: self.refresh_gt_list())

        cols = ("key", "boxes", "req", "reviewed", "score", "status")
        self.tree = ttk.Treeview(listf, columns=cols, show="headings", height=18)
        for c, t, w in [("key", "图片", 180), ("boxes", "框", 40),
                        ("req", "必", 35), ("reviewed", "审", 40),
                        ("score", "F1", 50), ("status", "检测", 55)]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center" if c != "key" else "w")
        sb = ttk.Scrollbar(listf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        self.tree.tag_configure("pass", foreground="#008800")
        self.tree.tag_configure("fail", foreground="#cc0000")
        self.tree.tag_configure("untested", foreground="#666666")
        self.tree.tag_configure("unreviewed", background="#fff5e0")
        self.tree.bind('<Double-Button-1>', self._on_gt_double)

        lbtn = ttk.Frame(left); lbtn.pack(fill=tk.X, pady=4)
        ttk.Button(lbtn, text="移除选中", command=self.remove_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(lbtn, text="清空库", command=self.clear_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(lbtn, text="刷新", command=self.refresh_gt_list).pack(side=tk.LEFT, padx=4)
        main.add(left, weight=2)

        right = ttk.Frame(main)
        logf = ttk.LabelFrame(right, text="学习日志")
        logf.pack(fill=tk.BOTH, expand=True)
        sb2 = ttk.Scrollbar(logf); sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.log = tk.Text(logf, wrap=tk.WORD, yscrollcommand=sb2.set,
                            font=('Consolas', 10), bg='#1e1e1e', fg='#d4d4d4')
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb2.config(command=self.log.yview)
        self.log.config(state=tk.DISABLED)
        self.log.tag_config("ok", foreground="#00ff88")
        self.log.tag_config("warn", foreground="#ffcc00")
        self.log.tag_config("err", foreground="#ff6666")
        self.log.tag_config("title", foreground="#66ccff", font=('Consolas', 11, 'bold'))
        self.log.tag_config("dim", foreground="#888888")
        main.add(right, weight=5)

        ttk.Label(self, text="提示：对比窗口右键（或点选）标记多余；完美训练耗时长，请谨慎使用",
                  foreground="#666").pack(anchor=tk.W, padx=10, pady=4)

        self.after(100, self._refresh_intensive_btn)

    # ==================== GT 列表 ====================
    def refresh_gt_list(self):
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            gt = self.app.ground_truth
            filt = self.gt_filter_var.get()
            for key, rec in gt.all_items():
                reviewed = rec.get("reviewed", False)
                status = rec.get("status", "untested")
                score = rec.get("last_score")

                if filt == "已审核" and not reviewed: continue
                if filt == "待审核" and reviewed: continue

                # ===== 状态只看 F1 =====
                if status == "img_missing":
                    st_txt = "图片缺失"
                    tag_status = "fail"
                elif score is None:
                    st_txt = "-"
                    tag_status = "untested"
                elif score >= PASS_F1_THR - 1e-6:
                    st_txt = "✓通过"
                    tag_status = "pass"
                else:
                    st_txt = "✗未通过"
                    tag_status = "fail"

                if filt == "未通过" and tag_status != "fail": continue
                if filt == "已通过" and tag_status != "pass": continue

                score_str = f"{score:.3f}" if score is not None else "----"
                rev_txt = "✓" if reviewed else "⚠"
                req_n = sum(1 for b in rec.get("boxes", []) if b.get("required", False))
                tags = (tag_status,)
                if not reviewed: tags = tags + ("unreviewed",)
                self.tree.insert("", "end", iid=key,
                    values=(key, len(rec.get("boxes", [])), req_n, rev_txt, score_str,
                            st_txt), tags=tags)
            self.stat_var.set(
                f"GT 总数: {gt.count()}  |  已审核: {gt.count_reviewed()}  |  "
                f"待审核: {gt.count_unreviewed()}")
        except Exception as e:
            print(f"[GT列表刷新失败] {e}")

    def _on_gt_double(self, event):
        sel = self.tree.selection()
        if not sel: return
        key = sel[0]
        before_m = after_m = None
        if self._last_report:
            before_m = self._last_report.get("before", {}).get(key)
            after_m = self._last_report.get("after", {}).get(key)
        self._open_compare_window(key, before_m, after_m)

    def remove_selected(self):
        sel = self.tree.selection()
        if not sel: messagebox.showinfo("提示", "请先选一条"); return
        key = sel[0]
        if messagebox.askyesno("确认", f"移除?\n{key}"):
            self.app.ground_truth.remove(key)
            self._refresh_all_gt_views()

    def clear_all(self):
        if not self.app.ground_truth.count(): return
        if messagebox.askyesno("确认", f"清空整个 GT 库（{self.app.ground_truth.count()} 张）？"):
            for k in list(self.app.ground_truth.data.keys()):
                self.app.ground_truth.remove(k)
            self._refresh_all_gt_views()

    # ==================== 只考试一次 ====================
    def review_once(self):
        if self.app.detector is None:
            messagebox.showwarning("提示", "请先在 GT 管理页加载模型"); return
        reviewed = self.app.ground_truth.all_reviewed_items()
        if not reviewed:
            messagebox.showinfo("提示", "没有已审核的 GT"); return
        if self.running:
            messagebox.showinfo("提示", "学习循环正在运行"); return
        self.stop_flag = False
        self._log_msg(f"开始考试（已审核 {len(reviewed)} 张）...", "title")
        self._set_progress(0, "准备考试...")
        threading.Thread(target=self._do_once, daemon=True).start()

    def _do_once(self):
        try:
            result = self._evaluate_all(progress_range=(0, 100), save_preds=True)
            pass_n = sum(1 for v in result.values()
                         if v.get("f1", 0) >= PASS_F1_THR - 1e-6)
            fail_n = len(result) - pass_n
            gt = self.app.ground_truth
            for key, m in result.items():
                st = "pass" if m.get("f1", 0) >= PASS_F1_THR - 1e-6 else "fail"
                gt.update_score(key, m["f1"], st)
            self.after(0, self._refresh_all_gt_views)
            self._log_msg(f"考试完成: 通过 {pass_n}, 未通过 {fail_n}", "title")
            analysis = self._analyze_errors(result)
            self._print_analysis(analysis)
            self._last_report = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "before": result, "after": result,
                "learned": 0, "still_fail": fail_n, "forgot": 0, "kept": pass_n,
                "loss_weights": analysis, "rolled_back": False,
            }
            self.after(0, lambda: self._log_msg("报告已生成，点「学习报告」查看", "ok"))
        except Exception:
            self._log_msg(f"异常: {traceback.format_exc()}", "err")
        finally:
            self.stop_flag = False
            self._set_progress(0, "就绪")

    # ==================== 闭环 ====================
    def review_loop(self):
        if self.app.detector is None:
            messagebox.showwarning("提示", "请先加载模型"); return
        reviewed = self.app.ground_truth.all_reviewed_items()
        if not reviewed:
            messagebox.showinfo("提示", "没有已审核的 GT"); return
        if self.running:
            messagebox.showinfo("提示", "闭环已在运行"); return
        self.running = True; self.stop_flag = False
        self.btn_once.config(state=tk.DISABLED)
        self.btn_loop.config(state=tk.DISABLED)
        self.btn_intensive.config(state=tk.DISABLED)
        self.btn_perfect.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self._do_loop, daemon=True).start()

    def stop_loop(self):
        self.stop_flag = True
        self._log_msg("收到停止请求...", "warn")
        self.progress_text.set("正在停止...")
        self.btn_stop.config(state=tk.DISABLED)

    def _do_loop(self):
        try:
            max_it = int(self.max_iter.get())
            epochs = int(self.epochs_per_iter.get())
            decay = float(self.lr_decay.get())
            do_rollback = bool(self.auto_rollback.get())

            last_before = last_after = None
            last_learned = last_still = last_forgot = last_kept = 0
            last_analysis = None
            rolled_back = False

            for it in range(max_it):
                if self.stop_flag: break
                lr_this = 0.001 * (decay ** it)
                self._log_line("")
                self._log_msg(f"════════════ 迭代 {it+1}/{max_it} (lr={lr_this:.6f}) ════════════", "title")
                self._log_msg("【步骤 1/4】看标准答案库", "title")
                gt = self.app.ground_truth
                items = gt.all_reviewed_items()
                self._log_msg(f"  共 {len(items)} 张已审核 GT", "dim")
                self._set_progress(0, f"迭代 {it+1}/{max_it} 准备中")
                if self.stop_flag: break

                self._log_msg("【步骤 2/4】学习前摸底考试", "title")
                before = self._evaluate_all(progress_range=(0, 30), save_preds=True)
                pass_before = sum(1 for v in before.values()
                                  if v.get("f1", 0) >= PASS_F1_THR - 1e-6)
                fail_before = len(before) - pass_before
                avg_before = (sum(v["f1"] for v in before.values()) / len(before)) if before else 0
                self._log_msg(
                    f"  学习前: 通过 {pass_before}/{len(before)}, 平均F1={avg_before:.4f}",
                    "ok" if fail_before == 0 else "warn")
                if fail_before == 0:
                    self._log_msg("  已全部通过，无需学习", "ok")
                    last_before = before; last_after = before; last_kept = pass_before
                    break
                if self.stop_flag: break

                analysis = self._analyze_errors(before)
                self._print_analysis(analysis)
                last_analysis = analysis

                self._log_msg("【步骤 3/4】用标准答案微调", "title")
                exported = self._export_all_gt_to_training(before_results=before)
                self._log_msg(f"  已导出 {exported} 张", "dim")
                if exported == 0:
                    self._log_msg("  没有可学习的图，终止", "err"); break

                ok = self._run_finetune(epochs, progress_range=(30, 70),
                                        loss_weights=analysis, lr0=lr_this)
                if not ok:
                    self._log_msg("  学习失败，终止", "err"); break
                if not self._reload_model(): break
                if self.stop_flag: break

                self._log_msg("【步骤 4/4】学习后对比考试", "title")
                after = self._evaluate_all(progress_range=(70, 100), save_preds=True)
                pass_after = sum(1 for v in after.values()
                                 if v.get("f1", 0) >= PASS_F1_THR - 1e-6)
                fail_after = len(after) - pass_after
                avg_after = (sum(v["f1"] for v in after.values()) / len(after)) if after else 0
                self._log_msg(
                    f"  学习后: 通过 {pass_after}/{len(after)}, 平均F1={avg_after:.4f}",
                    "ok" if fail_after == 0 else "warn")

                if do_rollback and avg_after < avg_before:
                    self._log_msg(f"  学后变差，触发回退", "err")
                    self._rollback_from_backup()
                    rolled_back = True
                    last_before = before; last_after = before; last_kept = pass_before
                    break

                learned = still_fail = forgot = kept = 0
                for key in before:
                    b = before[key]; a = after.get(key, {})
                    b_pass = b.get("f1", 0) >= PASS_F1_THR - 1e-6
                    a_pass = a.get("f1", 0) >= PASS_F1_THR - 1e-6
                    if a_pass and not b_pass: learned += 1
                    elif not a_pass and not b_pass: still_fail += 1
                    elif not a_pass and b_pass: forgot += 1
                    else: kept += 1

                for key, m in after.items():
                    st = "pass" if m.get("f1", 0) >= PASS_F1_THR - 1e-6 else "fail"
                    gt.update_score(key, m["f1"], st)
                self.after(0, self._refresh_all_gt_views)

                last_before = before; last_after = after
                last_learned = learned; last_still = still_fail
                last_forgot = forgot; last_kept = kept

                if fail_after == 0:
                    self._log_msg("全部通过！学习完成", "ok"); break

            if last_before is not None and last_after is not None:
                self._last_report = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "before": last_before, "after": last_after,
                    "learned": last_learned, "still_fail": last_still,
                    "forgot": last_forgot, "kept": last_kept,
                    "loss_weights": last_analysis, "rolled_back": rolled_back,
                }
                self._log_msg("学习报告已生成", "ok")
            self._log_msg("闭环结束", "title")
        except Exception:
            self._log_msg(f"闭环异常: {traceback.format_exc()}", "err")
        finally:
            self.running = False; self.stop_flag = False
            self.after(0, lambda: self.btn_once.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_loop.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_intensive.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_perfect.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))
            self.after(0, lambda: self._set_progress(0, "就绪"))
            self.after(0, self._refresh_intensive_btn)

    # ==================== 智能杂交：改为打开模型浏览器 ====================
    def open_model_browser(self):
        try:
            from model_finder import open_model_finder
            open_model_finder(self, self.app)
            self._log_msg("已打开模型浏览器；请选中 ≥2 个模型 → 点「🧬 智能杂交」", "dim")
        except Exception as e:
            messagebox.showerror("打开失败", f"{type(e).__name__}: {e}")

    # ==================== 保存 / 另存为 ====================
    def save_current_model(self):
        src = Path(self.app.weights_pred.get())
        if not src.exists():
            messagebox.showerror("失败", f"当前模型文件不存在:\n{src}")
            return
        bd = PROJECT_ROOT / "results" / "best_backups"
        bd.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = bd / f"{src.stem}_{ts}{src.suffix}"
        try:
            shutil.copy2(src, dst)
            self._log_msg(f"已保存当前模型: {dst.name}", "ok")
            messagebox.showinfo("已保存", f"已保存到:\n{dst}")
        except Exception as e:
            self._log_msg(f"保存失败: {e}", "err")
            messagebox.showerror("失败", f"{type(e).__name__}: {e}")

    def save_current_model_as(self):
        src = Path(self.app.weights_pred.get())
        if not src.exists():
            messagebox.showerror("失败", f"当前模型文件不存在:\n{src}")
            return
        default_dir = PROJECT_ROOT / "results" / "soup"
        default_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{src.stem}_{ts}.pt"
        p = filedialog.asksaveasfilename(
            title="另存为",
            initialdir=str(default_dir),
            initialfile=default_name,
            defaultextension=".pt",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if not p:
            return
        try:
            shutil.copy2(src, p)
            self._log_msg(f"已另存为: {p}", "ok")
            messagebox.showinfo("已保存", f"已保存到:\n{p}")
        except Exception as e:
            self._log_msg(f"另存为失败: {e}", "err")
            messagebox.showerror("失败", f"{type(e).__name__}: {e}")