# model_finder.py
# 模型浏览器：扫描所有 .pt，读取训练指标，支持排序/筛选/评估/切换/删除/批量排名/智能杂交/高亮标记
# v1.4:
#   - ⭐ 高亮标记 + 只看标记
#   - 去掉自定义全屏按钮，改用系统最大化
#   - SoupDialog 也去掉自定义全屏，用系统最大化
import json
import math
import os
import shutil
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image

from core import PROJECT_ROOT, evaluate_one


class ModelMarks:
    """模型高亮标记：⭐ 收藏"""
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
    def _key(p):
        try:
            return str(Path(p).resolve())
        except Exception:
            return str(p)

    def is_marked(self, p):
        return self._key(p) in self.data

    def mark(self, p, note=""):
        self.data[self._key(p)] = {
            "marked_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
        }
        self.save()

    def unmark(self, p):
        k = self._key(p)
        if k in self.data:
            del self.data[k]
            self.save()

    def toggle(self, p):
        if self.is_marked(p):
            self.unmark(p)
            return False
        else:
            self.mark(p)
            return True

    def count(self):
        return len(self.data)


class ModelFinderWindow:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.records = []
        self._iid_to_rec = {}
        self._sort_key = "mtime"
        self._sort_rev = True
        self._eval_running = False
        self._eval_cancel = False
        self._rank_mode = False
        self.marks = ModelMarks(PROJECT_ROOT / "results" / "model_marks.json")

        self.win = tk.Toplevel(parent)
        self.win.title("模型浏览器 - 找模型 / 批量筛选 / 智能杂交 / 高亮标记")
        self.win.geometry("1360x820")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.win.bind("<Escape>", lambda e: self._on_close())

        self._build_ui()
        self.refresh()

    # ==================== UI ====================
    def _build_ui(self):
        # 标题
        title_bar = ttk.Frame(self.win)
        title_bar.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Label(title_bar, text="🔍 模型浏览器",
                  font=('Consolas', 12, 'bold'),
                  foreground="#007acc").pack(side=tk.LEFT)
        ttk.Label(title_bar, text="（ESC 关闭；标记后可用「只看标记」快速定位）",
                  font=('Consolas', 9), foreground="#888").pack(side=tk.LEFT, padx=8)

        info = ttk.LabelFrame(self.win, text="说明")
        info.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Label(
            info,
            text="双击行 → 用这个模型；「评估选中」→ 用当前 GT 库实测 F1。\n"
                 "「🏆 批量筛选」→ 一次评估多个模型并按 GT-F1 排名。\n"
                 "「🧬 智能杂交」→ 选中 ≥2 个模型，搜索最优融合权重。\n"
                 "「⭐ 标记」→ 给好模型打星，筛选「只看标记」时只显示这些。",
            justify=tk.LEFT, font=('Consolas', 10)
        ).pack(anchor=tk.W, padx=8, pady=6)

        # 筛选
        top = ttk.Frame(self.win)
        top.pack(fill=tk.X, padx=10, pady=4)

        ttk.Label(top, text="筛选:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.filter_var, width=24).pack(side=tk.LEFT, padx=6)
        self.filter_var.trace_add('write', lambda *a: self._refill())

        self.only_marked_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="⭐ 只看标记",
                        variable=self.only_marked_var,
                        command=self._refill).pack(side=tk.LEFT, padx=(4, 8))

        ttk.Label(top, text="最低 mAP50:").pack(side=tk.LEFT)
        self.min_map_var = tk.StringVar(value="0.0")
        ttk.Entry(top, textvariable=self.min_map_var, width=6).pack(side=tk.LEFT, padx=4)
        self.min_map_var.trace_add('write', lambda *a: self._refill())

        ttk.Label(top, text="来源:").pack(side=tk.LEFT, padx=(10, 2))
        self.source_var = tk.StringVar(value="全部")
        ttk.Combobox(top, textvariable=self.source_var, width=12, state="readonly",
                     values=["全部", "exp", "learn", "perfect", "intensive",
                             "batch_ft", "backup", "root", "soup"]).pack(side=tk.LEFT, padx=4)
        self.source_var.trace_add('write', lambda *a: self._refill())

        ttk.Button(top, text="重新扫描", command=self.refresh).pack(side=tk.RIGHT, padx=4)

        # 操作按钮
        ops = ttk.Frame(self.win)
        ops.pack(fill=tk.X, padx=10, pady=(0, 4))

        self.btn_eval = ttk.Button(ops, text="评估选中", command=self.evaluate_selected)
        self.btn_eval.pack(side=tk.LEFT, padx=3)

        self.btn_screen = ttk.Button(ops, text="🏆 批量筛选(测试+排名)",
                                     command=self.batch_screen)
        self.btn_screen.pack(side=tk.LEFT, padx=3)

        self.btn_soup = ttk.Button(ops, text="🧬 智能杂交",
                                   command=self._open_soup_dialog)
        self.btn_soup.pack(side=tk.LEFT, padx=3)

        self.btn_cancel = ttk.Button(ops, text="停止", command=self._cancel_eval,
                                     state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=3)

        ttk.Separator(ops, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_mark = ttk.Button(ops, text="⭐ 标记选中",
                                   command=self.toggle_mark_selected)
        self.btn_mark.pack(side=tk.LEFT, padx=3)

        self.btn_use = ttk.Button(ops, text="用这个模型", command=self.use_selected)
        self.btn_use.pack(side=tk.LEFT, padx=3)

        self.btn_backup = ttk.Button(ops, text="备份选中", command=self.backup_selected)
        self.btn_backup.pack(side=tk.LEFT, padx=3)

        self.btn_delete = ttk.Button(ops, text="🗑️ 删除选中", command=self.delete_selected)
        self.btn_delete.pack(side=tk.LEFT, padx=3)

        ttk.Separator(ops, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(ops, text="全选", command=self.select_all).pack(side=tk.LEFT, padx=3)
        ttk.Button(ops, text="反选", command=self.invert_sel).pack(side=tk.LEFT, padx=3)
        ttk.Button(ops, text="取消选择", command=self.clear_sel).pack(side=tk.LEFT, padx=3)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text = tk.StringVar(value="")
        ttk.Progressbar(ops, variable=self.progress_var, maximum=100,
                        length=180).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Label(ops, textvariable=self.progress_text,
                  font=('Consolas', 9), foreground="#cc6600").pack(side=tk.RIGHT, padx=4)

        # 表格
        listf = ttk.LabelFrame(self.win, text="模型列表（双击切换，可多选；右键标记）")
        listf.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        cols = ("mark", "rank", "name", "source", "mtime", "size",
                "mAP50", "P", "R", "F1", "gt_f1", "path")
        self.tree = ttk.Treeview(listf, columns=cols, show="headings",
                                 selectmode="extended", height=20)
        headers = [
            ("mark",   "⭐",      30, "center"),
            ("rank",   "排名",    55, "center"),
            ("name",   "文件名",  200, "w"),
            ("source", "来源",    90,  "center"),
            ("mtime",  "修改时间", 150, "center"),
            ("size",   "大小MB",  70,  "center"),
            ("mAP50",  "mAP50",   70,  "center"),
            ("P",      "P",       60,  "center"),
            ("R",      "R",       60,  "center"),
            ("F1",     "csv-F1",  70,  "center"),
            ("gt_f1",  "GT-F1",   80,  "center"),
            ("path",   "路径",    360, "w"),
        ]
        for c, t, w, anchor in headers:
            self.tree.heading(c, text=t, command=lambda _c=c: self._sort_by(_c))
            self.tree.column(c, width=w, anchor=anchor)

        sb = ttk.Scrollbar(listf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # 标记：金色背景
        self.tree.tag_configure("marked", background="#fff0a0")
        self.tree.tag_configure("rank1", foreground="#cc0066", font=('Consolas', 10, 'bold'))
        self.tree.tag_configure("best", foreground="#008800", font=('Consolas', 10, 'bold'))
        self.tree.tag_configure("perfect", foreground="#cc0066")
        self.tree.tag_configure("current", background="#fff5cc")
        self.tree.tag_configure("pretrained", foreground="#888888")
        self.tree.tag_configure("soup", foreground="#0066cc")

        self.tree.bind("<Double-Button-1>", self._on_double)
        self.tree.bind("<Control-a>", lambda e: (self.select_all(), "break"))
        self.tree.bind("<Delete>", lambda e: self.delete_selected())
        # 右键菜单
        self.tree.bind("<Button-3>", self._on_right_click)

        # 右键菜单
        self.menu = tk.Menu(self.win, tearoff=0)
        self.menu.add_command(label="⭐ 标记 / 取消标记", command=self.toggle_mark_selected)
        self.menu.add_separator()
        self.menu.add_command(label="用这个模型", command=self.use_selected)
        self.menu.add_command(label="备份选中", command=self.backup_selected)
        self.menu.add_command(label="复制路径", command=self.copy_path)
        self.menu.add_command(label="打开所在目录", command=self.open_dir)
        self.menu.add_separator()
        self.menu.add_command(label="🗑️ 删除选中", command=self.delete_selected)

        # 底部
        bottom = ttk.Frame(self.win)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.stat_var = tk.StringVar(value="共 0 个")
        ttk.Label(bottom, textvariable=self.stat_var,
                  font=('Consolas', 10), foreground="#007acc").pack(side=tk.LEFT)

        self.mark_stat_var = tk.StringVar(value="⭐ 已标记: 0")
        ttk.Label(bottom, textvariable=self.mark_stat_var,
                  font=('Consolas', 10), foreground="#cc8800").pack(side=tk.LEFT, padx=(16, 0))

        ttk.Button(bottom, text="关闭", command=self._on_close).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="打开所在目录",
                   command=self.open_dir).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="复制路径",
                   command=self.copy_path).pack(side=tk.RIGHT, padx=4)

    # ==================== 右键 ====================
    def _on_right_click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid:
            if iid not in self.tree.selection():
                self.tree.selection_set(iid)
            try:
                self.menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.menu.grab_release()

    # ==================== 扫描 ====================
    def refresh(self):
        self.records = []
        self.records.extend(self._scan_runs())
        self.records.extend(self._scan_backups())
        self.records.extend(self._scan_root())
        self.records.extend(self._scan_soup_outputs())

        seen = set()
        uniq = []
        for r in self.records:
            key = str(Path(r["path"]).resolve())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        self.records = uniq

        self._rank_mode = False
        for r in self.records:
            r["rank"] = None

        self._refill()
        self.stat_var.set(f"共 {len(self.records)} 个模型")
        self.mark_stat_var.set(f"⭐ 已标记: {self.marks.count()}")

    def _scan_runs(self):
        out = []
        runs_dir = PROJECT_ROOT / "runs"
        if not runs_dir.exists():
            return out
        for pt in runs_dir.rglob("*.pt"):
            if pt.parent.name != "weights" and pt.name not in ("best.pt", "last.pt"):
                continue
            out.append(self._make_record(pt, source=self._guess_source(pt)))
        return out

    def _scan_backups(self):
        out = []
        bd = PROJECT_ROOT / "results" / "best_backups"
        if not bd.exists():
            return out
        for pt in bd.glob("*.pt"):
            out.append(self._make_record(pt, source="backup"))
        return out

    def _scan_root(self):
        out = []
        pretrained_names = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt",
                            "yolo11x.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt")
        for pt in PROJECT_ROOT.glob("*.pt"):
            is_pretrain = pt.name in pretrained_names
            out.append(self._make_record(pt, source="root", is_pretrained=is_pretrain))
        return out

    def _scan_soup_outputs(self):
        out = []
        sd = PROJECT_ROOT / "results" / "soup"
        if not sd.exists():
            return out
        for pt in sd.rglob("*.pt"):
            rec = self._make_record(pt, source="soup")
            out.append(rec)
        return out

    @staticmethod
    def _guess_source(pt):
        p = str(pt).replace("\\", "/")
        if "/soup/" in p: return "soup"
        if "/learn_" in p: return "learn"
        if "/perfect_" in p: return "perfect"
        if "/int_" in p: return "intensive"
        if "/batch_ft_" in p: return "batch_ft"
        if "/exp" in p: return "exp"
        return "exp"

    def _make_record(self, pt, source="exp", is_pretrained=False):
        try:
            st = pt.stat()
            mtime = datetime.fromtimestamp(st.st_mtime)
            size_mb = st.st_size / 1024 / 1024
        except Exception:
            mtime = datetime.now()
            size_mb = 0.0

        mAP50 = P = R = F1 = None
        if not is_pretrained:
            metrics = self._read_results_csv(pt)
            if metrics:
                mAP50 = metrics.get("mAP50")
                P = metrics.get("P")
                R = metrics.get("R")
                if P is not None and R is not None and (P + R) > 0:
                    F1 = 2 * P * R / (P + R)

        return {
            "name": pt.name,
            "source": source,
            "mtime": mtime.strftime("%Y-%m-%d %H:%M:%S"),
            "mtime_obj": mtime,
            "size": size_mb,
            "mAP50": mAP50,
            "P": P,
            "R": R,
            "F1": F1,
            "gt_f1": None,
            "rank": None,
            "path": str(pt),
            "is_pretrained": is_pretrained,
            "marked": self.marks.is_marked(str(pt)),
        }

    @staticmethod
    def _read_results_csv(pt):
        try:
            run_dir = pt.parent.parent
            csv = run_dir / "results.csv"
            if not csv.exists():
                return None
            with open(csv, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if len(lines) < 2:
                return None
            headers = [h.strip() for h in lines[0].split(",")]
            last = lines[-1].split(",")
            out = {}
            for i, h in enumerate(headers):
                if i >= len(last):
                    break
                h_low = h.strip().lower()
                try:
                    val = float(last[i])
                except Exception:
                    continue
                if "map50" in h_low and "95" not in h_low:
                    out["mAP50"] = val
                elif h_low.startswith("metrics/precision"):
                    out["P"] = val
                elif h_low.startswith("metrics/recall"):
                    out["R"] = val
            return out if out else None
        except Exception:
            return None

    # ==================== 表格 ====================
    def _sort_by(self, key):
        if self._sort_key == key:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_key = key
            self._sort_rev = True
        self._refill()

    def _refill(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._iid_to_rec.clear()

        kw = self.filter_var.get().strip().lower()
        try:
            min_map = float(self.min_map_var.get())
        except Exception:
            min_map = 0.0
        src = self.source_var.get()
        only_marked = self.only_marked_var.get()

        current_w = str(Path(self.app.weights_pred.get()).resolve()) \
            if self.app.weights_pred.get() else ""

        rows = []
        for r in self.records:
            # 同步标记状态
            r["marked"] = self.marks.is_marked(r["path"])
            if only_marked and not r["marked"]:
                continue
            if kw and kw not in r["name"].lower() and kw not in r["path"].lower():
                continue
            if src != "全部" and r["source"] != src:
                continue
            if r["mAP50"] is not None and r["mAP50"] < min_map:
                continue
            rows.append(r)

        # 排序：标记的永远排前面
        def marked_key(r):
            return 0 if r.get("marked") else 1

        if self._rank_mode:
            rows.sort(key=lambda r: (marked_key(r),
                                      r.get("rank") is None,
                                      r.get("rank") or 9999))
        else:
            try:
                if self._sort_key in ("mAP50", "F1", "gt_f1", "size", "rank"):
                    def kf(r):
                        v = r.get(self._sort_key)
                        if v is None:
                            return (marked_key(r), 1, 0)
                        return (marked_key(r), 0, -v if self._sort_rev else v)
                    rows.sort(key=kf)
                elif self._sort_key == "mtime":
                    rows.sort(key=lambda r: (marked_key(r), -r["mtime_obj"].timestamp()
                                              if self._sort_rev
                                              else r["mtime_obj"].timestamp()))
                else:
                    rows.sort(key=lambda r: (marked_key(r),
                                              r.get(self._sort_key) or ""),
                              reverse=self._sort_rev)
            except Exception:
                pass

        for r in rows:
            tags = []
            if r.get("marked"):
                tags.append("marked")
            if r.get("rank") == 1:
                tags.append("rank1")
            if r["mAP50"] is not None and r["mAP50"] >= 0.995:
                tags.append("perfect")
            if r["F1"] is not None and r["F1"] >= 0.999:
                tags.append("best")
            if r.get("is_pretrained"):
                tags.append("pretrained")
            if r.get("source") == "soup":
                tags.append("soup")
            try:
                if current_w and str(Path(r["path"]).resolve()) == current_w:
                    tags.append("current")
            except Exception:
                pass

            def fmt(v, n=4):
                return f"{v:.{n}f}" if isinstance(v, (int, float)) else "-"

            rank_txt = f"#{r['rank']}" if r.get("rank") else "-"
            mark_txt = "⭐" if r.get("marked") else ""

            iid = self.tree.insert(
                "", "end",
                values=(mark_txt, rank_txt, r["name"], r["source"], r["mtime"],
                        f"{r['size']:.1f}", fmt(r["mAP50"]), fmt(r["P"]),
                        fmt(r["R"]), fmt(r["F1"]), fmt(r["gt_f1"]), r["path"]),
                tags=tuple(tags))
            self._iid_to_rec[iid] = r

        self.stat_var.set(f"共 {len(rows)} / {len(self.records)} 个模型")
        self.mark_stat_var.set(f"⭐ 已标记: {self.marks.count()}")

    def _selected_records(self):
        out = []
        for iid in self.tree.selection():
            r = self._iid_to_rec.get(iid)
            if r is not None:
                out.append(r)
        return out

    def _visible_records(self):
        return [self._iid_to_rec[iid] for iid in self.tree.get_children()
                if iid in self._iid_to_rec]

    # ==================== 选择 ====================
    def select_all(self):
        self.tree.selection_set(self.tree.get_children())

    def invert_sel(self):
        selected = set(self.tree.selection())
        all_ids = set(self.tree.get_children())
        self.tree.selection_remove(*all_ids)
        for iid in (all_ids - selected):
            self.tree.selection_add(iid)

    def clear_sel(self):
        self.tree.selection_remove(*self.tree.selection())

    # ==================== 标记 ====================
    def toggle_mark_selected(self):
        recs = self._selected_records()
        if not recs:
            messagebox.showinfo("提示", "请先选模型")
            return
        n_marked = 0
        n_unmarked = 0
        for r in recs:
            if self.marks.is_marked(r["path"]):
                self.marks.unmark(r["path"])
                n_unmarked += 1
            else:
                self.marks.mark(r["path"])
                n_marked += 1
        self._refill()
        self.mark_stat_var.set(f"⭐ 已标记: {self.marks.count()}")
        self.app.status.set(f"标记 {n_marked} 个，取消 {n_unmarked} 个")

    # ==================== 操作 ====================
    def _on_double(self, event):
        recs = self._selected_records()
        if not recs:
            return
        self._switch_to(recs[0])

    def use_selected(self):
        recs = self._selected_records()
        if not recs:
            messagebox.showinfo("提示", "请先选一个模型")
            return
        if len(recs) > 1:
            if not messagebox.askyesno("确认", f"选中 {len(recs)} 个，只用第一个？"):
                return
        self._switch_to(recs[0])

    def _switch_to(self, rec):
        if rec.get("is_pretrained"):
            if not messagebox.askyesno("确认",
                    f"{rec['name']} 是预训练权重，确定切换？"):
                return
        if not messagebox.askyesno("切换模型",
                f"用这个模型？\n\n{rec['name']}\n{rec['path']}"):
            return
        self.app.weights_pred.set(rec["path"])
        try:
            from predict import Detector
            self.app.detector = Detector(rec["path"])
            if hasattr(self.app, "gt_tab"):
                self.app.gt_tab.refresh_model_info()
            self.app.status.set(f"已切换模型: {rec['name']}")
            messagebox.showinfo("成功", f"已切换到:\n{rec['name']}")
            self._refill()
        except Exception as e:
            messagebox.showerror("失败", f"{type(e).__name__}: {e}")

    def backup_selected(self):
        recs = self._selected_records()
        if not recs:
            messagebox.showinfo("提示", "请先选一个模型")
            return
        bd = PROJECT_ROOT / "results" / "best_backups"
        bd.mkdir(parents=True, exist_ok=True)
        ok = 0
        fail = []
        for r in recs:
            try:
                src = Path(r["path"])
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst = bd / f"{src.stem}_{ts}{src.suffix}"
                shutil.copy2(src, dst)
                ok += 1
            except Exception as e:
                fail.append(f"{r['name']}: {e}")
        msg = f"已备份 {ok} 个到:\n{bd}"
        if fail:
            msg += f"\n\n失败 {len(fail)} 个:\n" + "\n".join(fail[:5])
        messagebox.showinfo("完成", msg)
        self.refresh()

    def copy_path(self):
        recs = self._selected_records()
        if not recs:
            return
        text = "\n".join(r["path"] for r in recs)
        self.win.clipboard_clear()
        self.win.clipboard_append(text)
        messagebox.showinfo("已复制", f"已复制 {len(recs)} 条路径到剪贴板")

    def open_dir(self):
        recs = self._selected_records()
        if not recs:
            messagebox.showinfo("提示", "请先选一个模型")
            return
        d = Path(recs[0]["path"]).parent
        try:
            os.startfile(str(d))
        except Exception as e:
            messagebox.showerror("失败", f"{e}")

    # ==================== 删除 ====================
    def delete_selected(self):
        recs = self._selected_records()
        if not recs:
            messagebox.showinfo("提示", "请先选要删除的模型")
            return

        current_w = str(Path(self.app.weights_pred.get()).resolve()) \
            if self.app.weights_pred.get() else ""
        warnings = []
        pretrained_n = 0
        marked_n = 0
        for r in recs:
            try:
                if current_w and str(Path(r["path"]).resolve()) == current_w:
                    warnings.append(f"⚠ {r['name']} 是当前正在使用的模型！")
            except Exception:
                pass
            if r.get("is_pretrained"):
                pretrained_n += 1
            if self.marks.is_marked(r["path"]):
                marked_n += 1
        if pretrained_n:
            warnings.append(f"⚠ 其中有 {pretrained_n} 个是预训练权重（yolo11n.pt 等）")
        if marked_n:
            warnings.append(f"⚠ 其中有 {marked_n} 个已被 ⭐ 标记")

        msg = f"确定删除 {len(recs)} 个模型文件？\n\n"
        for r in recs[:12]:
            msg += f"  · {r['name']}\n    {r['path']}\n"
        if len(recs) > 12:
            msg += f"  ...还有 {len(recs) - 12} 个\n"
        if warnings:
            msg += "\n" + "\n".join(warnings) + "\n"
        msg += "\n⚠ 文件将被永久删除（不进回收站），无法撤销！\n是否继续？"

        if not messagebox.askyesno("⚠ 确认删除", msg):
            return

        if len(recs) >= 5:
            if not messagebox.askyesno(
                "二次确认",
                f"再次确认：删除 {len(recs)} 个模型？"):
                return
            also_last = messagebox.askyesno(
                "同时删除 last.pt?",
                "是否同时删除同目录下的 last.pt？\n"
                "（一般不需要保留，除非要继续训练）")
        else:
            also_last = False

        deleted_pt = 0
        deleted_last = 0
        failed = []
        for r in recs:
            try:
                p = Path(r["path"])
                try:
                    if current_w and str(p.resolve()) == current_w:
                        self.app.weights_pred.set("")
                        self.app.detector = None
                except Exception:
                    pass

                p.unlink()
                deleted_pt += 1
                # 同步取消标记
                if self.marks.is_marked(str(p)):
                    self.marks.unmark(str(p))

                if also_last:
                    lp = p.parent / "last.pt"
                    if lp.exists() and lp != p:
                        try:
                            lp.unlink()
                            deleted_last += 1
                        except Exception:
                            pass
            except Exception as e:
                failed.append(f"{r['name']}: {e}")

        if hasattr(self.app, "gt_tab"):
            try:
                self.app.gt_tab.refresh_model_info()
            except Exception:
                pass

        tip = f"已删除 {deleted_pt} 个模型"
        if deleted_last:
            tip += f"\n同时删除 last.pt: {deleted_last} 个"
        if failed:
            tip += f"\n\n失败 {len(failed)} 个:\n" + "\n".join(failed[:5])
        messagebox.showinfo("完成", tip)
        self.refresh()

    # ==================== 单个评估 ====================
    def evaluate_selected(self):
        if self._eval_running:
            messagebox.showinfo("提示", "正在评估中")
            return
        recs = self._selected_records()
        if not recs:
            messagebox.showinfo("提示", "请先选一个模型")
            return
        reviewed = self.app.ground_truth.all_reviewed_items()
        if not reviewed:
            messagebox.showinfo("提示", "GT 库里没有已审核的图，先去「GT 库管理」审核")
            return
        self._start_eval(recs, reviewed, do_rank=False)

    # ==================== 批量筛选 ====================
    def batch_screen(self):
        if self._eval_running:
            messagebox.showinfo("提示", "正在评估中")
            return

        recs = self._selected_records()
        if not recs:
            recs = self._visible_records()
            if len(recs) == 0:
                messagebox.showinfo("提示", "没有可测试的模型")
                return
            if not messagebox.askyesno(
                "批量筛选",
                f"当前没有选中任何模型。\n"
                f"是否对当前筛选出的 {len(recs)} 个模型全部测试？"):
                return
        else:
            if not messagebox.askyesno(
                "批量筛选",
                f"对选中的 {len(recs)} 个模型逐一评估并按 GT-F1 排名？\n\n"
                f"注意：\n"
                f"  · 每个模型都会用 GT 库跑一遍推理\n"
                f"  · 模型多时会比较慢，可以随时「停止」\n"
                f"  · 预训练权重会自动跳过"):
                return

        recs = [r for r in recs if not r.get("is_pretrained")]
        if not recs:
            messagebox.showinfo("提示", "没有可评估的微调模型")
            return

        reviewed = self.app.ground_truth.all_reviewed_items()
        if not reviewed:
            messagebox.showinfo("提示", "GT 库里没有已审核的图，先去「GT 库管理」审核")
            return

        self._start_eval(recs, reviewed, do_rank=True)

    def _start_eval(self, recs, reviewed, do_rank=False):
        self._eval_running = True
        self._eval_cancel = False
        self._do_rank = do_rank
        self.btn_eval.config(state=tk.DISABLED)
        self.btn_screen.config(state=tk.DISABLED)
        self.btn_soup.config(state=tk.DISABLED)
        self.btn_delete.config(state=tk.DISABLED)
        self.btn_mark.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_text.set("准备...")
        self.app.status.set(f"评估 {len(recs)} 个模型...")

        threading.Thread(target=self._do_evaluate,
                         args=(recs, reviewed, do_rank), daemon=True).start()

    def _cancel_eval(self):
        self._eval_cancel = True
        self.progress_text.set("正在停止...")
        self.app.status.set("正在停止评估...")

    def _do_evaluate(self, recs, reviewed, do_rank=False):
        from predict import Detector
        total_m = len(recs)
        try:
            for mi, rec in enumerate(recs, 1):
                if self._eval_cancel:
                    break

                def set_prog(mi=mi, name=rec["name"]):
                    pct = (mi - 1) / max(total_m, 1) * 100
                    self.win.after(0, lambda: (
                        self.progress_var.set(pct),
                        self.progress_text.set(f"{mi}/{total_m} {name[:20]}")
                    ))
                set_prog()

                try:
                    detector = Detector(rec["path"])
                except Exception as e:
                    print(f"[评估] {rec['name']} 加载失败: {e}")
                    rec["gt_f1"] = 0.0
                    self.win.after(0, lambda r=rec: self._update_row(r))
                    continue

                total_f1 = 0.0
                n = 0
                for key, gt_rec in reviewed:
                    if self._eval_cancel:
                        break
                    img_path = self.app.ground_truth.img_dir / gt_rec["file"]
                    if not img_path.exists():
                        continue
                    try:
                        with Image.open(img_path) as im:
                            img_size = im.size
                    except Exception:
                        continue
                    try:
                        preds = detector.predict(str(img_path), conf=0.3)
                    except Exception:
                        continue
                    m = evaluate_one(gt_rec["boxes"], preds, iou_thr=0.5,
                                     img_size=img_size,
                                     use_area_weight=True,
                                     enforce_required=True)
                    total_f1 += m["f1"]
                    n += 1
                avg = total_f1 / n if n > 0 else 0.0
                rec["gt_f1"] = avg

                try:
                    del detector
                except Exception:
                    pass

                self.win.after(0, lambda r=rec: self._update_row(r))

            if not self._eval_cancel:
                if do_rank:
                    self._compute_ranks(recs)
                    self.win.after(0, lambda: self.progress_text.set("排名完成"))
                self.win.after(0, lambda: self.app.status.set("评估完成"))
            else:
                self.win.after(0, lambda: self.progress_text.set("已停止"))
        finally:
            self.win.after(0, self._eval_finished)

    def _compute_ranks(self, recs):
        evaluated = [r for r in recs if r.get("gt_f1") is not None]
        evaluated.sort(key=lambda r: -r["gt_f1"])
        for i, r in enumerate(evaluated, 1):
            r["rank"] = i
        for r in recs:
            if r.get("gt_f1") is None:
                r["rank"] = None

        all_evaluated = [r for r in self.records if r.get("gt_f1") is not None]
        all_evaluated.sort(key=lambda r: -r["gt_f1"])
        for i, r in enumerate(all_evaluated, 1):
            r["rank"] = i

        self._rank_mode = True

    def _eval_finished(self):
        self._eval_running = False
        self.btn_eval.config(state=tk.NORMAL)
        self.btn_screen.config(state=tk.NORMAL)
        self.btn_soup.config(state=tk.NORMAL)
        self.btn_delete.config(state=tk.NORMAL)
        self.btn_mark.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.progress_var.set(0 if not self._rank_mode else 100)
        if not self._rank_mode:
            self.progress_text.set("")

        if self._rank_mode:
            self._sort_key = "rank"
            self._sort_rev = False
        self._refill()

    def _update_row(self, rec):
        for iid, r in self._iid_to_rec.items():
            if r is rec or r["path"] == rec["path"]:
                gf = rec.get("gt_f1")
                gf_txt = f"{gf:.4f}" if gf is not None else "-"
                rank_txt = f"#{rec['rank']}" if rec.get("rank") else "-"
                mark_txt = "⭐" if self.marks.is_marked(rec["path"]) else ""
                vals = list(self.tree.item(iid, "values"))
                vals[0] = mark_txt
                vals[1] = rank_txt
                vals[-2] = gf_txt
                self.tree.item(iid, values=tuple(vals))
                tags = list(self.tree.item(iid, "tags"))
                if gf is not None and gf >= 0.999 and "best" not in tags:
                    tags.append("best")
                if gf is not None and gf >= 0.999 and "perfect" not in tags:
                    tags.append("perfect")
                self.tree.item(iid, tags=tuple(tags))
                break

    # ==================== 智能杂交 ====================
    def _open_soup_dialog(self):
        if self._eval_running:
            messagebox.showinfo("提示", "正在评估中，请稍候")
            return

        recs = [r for r in self._selected_records() if not r.get("is_pretrained")]
        if len(recs) < 2:
            visible = [r for r in self._visible_records()
                       if not r.get("is_pretrained")]
            if len(visible) < 2:
                messagebox.showinfo("提示", "请选中至少 2 个非预训练模型")
                return
            if not messagebox.askyesno(
                "智能杂交",
                f"当前只选中 {len(recs)} 个。\n"
                f"是否用当前筛选出的 {len(visible)} 个模型作为候选？"):
                return
            recs = visible

        reviewed = self.app.ground_truth.all_reviewed_items()
        if not reviewed:
            messagebox.showinfo("提示", "GT 库里没有已审核的图，无法评估候选模型")
            return

        SoupDialog(self, recs)

    # ==================== 关闭 ====================
    def _on_close(self):
        self._eval_cancel = True
        try:
            self.win.destroy()
        except Exception:
            pass


# ==================== 智能杂交对话框 ====================
class SoupDialog:
    def __init__(self, finder, candidates):
        self.finder = finder
        self.app = finder.app
        self.candidates = candidates
        self.running = False
        self.cancel_flag = False
        self.best_path = None

        self.win = tk.Toplevel(finder.win)
        self.win.title("🧬 智能杂交 - 搜索最优融合权重")
        self.win.geometry("1040x800")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.win.bind("<Escape>", lambda e: self._on_close())

        self._build_ui()

    def _build_ui(self):
        title_bar = ttk.Frame(self.win)
        title_bar.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Label(title_bar, text="🧬 智能杂交",
                  font=('Consolas', 12, 'bold'),
                  foreground="#007acc").pack(side=tk.LEFT)
        ttk.Label(title_bar, text="（ESC 关闭；点击标题栏方框可最大化）",
                  font=('Consolas', 9), foreground="#888").pack(side=tk.LEFT, padx=8)

        info = ttk.LabelFrame(self.win, text="说明")
        info.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Label(
            info,
            text="① 逐个评估候选模型 → ② 保留 top-K → ③ 贝叶斯搜索最优融合权重 → ④ 输出融合模型\n"
                 "候选模型来自你刚才在模型浏览器里选中的那批。",
            justify=tk.LEFT, font=('Consolas', 10)
        ).pack(anchor=tk.W, padx=8, pady=6)

        # ===== 底部按钮：先 pack，side=BOTTOM =====
        btns = ttk.Frame(self.win)
        btns.pack(fill=tk.X, padx=10, pady=8, side=tk.BOTTOM)
        self.btn_start = ttk.Button(btns, text="🚀 开始杂交", command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_stop = ttk.Button(btns, text="停止", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.btn_use = ttk.Button(btns, text="✅ 用融合模型", command=self.use_result,
                                  state=tk.DISABLED)
        self.btn_use.pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="关闭", command=self._on_close).pack(side=tk.RIGHT, padx=4)

        # ===== 进度条 =====
        prog = ttk.Frame(self.win)
        prog.pack(fill=tk.X, padx=10, pady=4, side=tk.BOTTOM)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text = tk.StringVar(value="就绪")
        ttk.Progressbar(prog, variable=self.progress_var, maximum=100,
                        length=500).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(prog, textvariable=self.progress_text,
                  font=('Consolas', 10), foreground="#cc6600").pack(side=tk.LEFT)

        # ===== 候选列表 =====
        listf = ttk.LabelFrame(self.win, text=f"候选模型 ({len(self.candidates)})")
        listf.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        cols = ("idx", "name", "mtime", "mAP50", "path")
        self.tree = ttk.Treeview(listf, columns=cols, show="headings", height=8)
        for c, t, w in [("idx", "#", 40), ("name", "文件名", 220),
                        ("mtime", "修改时间", 150), ("mAP50", "mAP50", 70),
                        ("path", "路径", 420)]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center" if c != "name" else "w")
        sb0 = ttk.Scrollbar(listf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb0.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        sb0.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        for i, r in enumerate(self.candidates, 1):
            mp = f"{r['mAP50']:.4f}" if r.get("mAP50") is not None else "-"
            self.tree.insert("", "end",
                             values=(i, r["name"], r["mtime"], mp, r["path"]))

        # ===== 参数 =====
        param = ttk.LabelFrame(self.win, text="搜索参数")
        param.pack(fill=tk.X, padx=10, pady=6)
        row = ttk.Frame(param); row.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(row, text="top_k:").pack(side=tk.LEFT)
        self.topk_var = tk.IntVar(value=min(5, len(self.candidates)))
        ttk.Spinbox(row, from_=2, to=10, textvariable=self.topk_var,
                    width=4).pack(side=tk.LEFT, padx=4)

        ttk.Label(row, text="搜索次数:").pack(side=tk.LEFT, padx=(12, 0))
        self.calls_var = tk.IntVar(value=30)
        ttk.Spinbox(row, from_=10, to=200, textvariable=self.calls_var,
                    width=5).pack(side=tk.LEFT, padx=4)

        ttk.Label(row, text="IoU:").pack(side=tk.LEFT, padx=(12, 0))
        self.iou_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(row, from_=0.3, to=0.9, increment=0.05,
                    textvariable=self.iou_var, width=5).pack(side=tk.LEFT, padx=4)

        out_row = ttk.Frame(param); out_row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(out_row, text="输出目录:").pack(side=tk.LEFT)
        default_out = PROJECT_ROOT / "results" / "soup"
        self.out_var = tk.StringVar(value=str(default_out))
        ttk.Entry(out_row, textvariable=self.out_var, width=68).pack(side=tk.LEFT, padx=4)
        ttk.Button(out_row, text="选择", command=self._choose_out).pack(side=tk.LEFT)

        # ===== 日志 =====
        logf = ttk.LabelFrame(self.win, text="搜索日志")
        logf.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        sb = ttk.Scrollbar(logf); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log = tk.Text(logf, wrap=tk.WORD, yscrollcommand=sb.set,
                           font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4',
                           height=12)
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb.config(command=self.log.yview)
        self.log.config(state=tk.DISABLED)
        self.log.tag_config("ok", foreground="#00ff88")
        self.log.tag_config("warn", foreground="#ffcc00")
        self.log.tag_config("err", foreground="#ff6666")
        self.log.tag_config("title", foreground="#66ccff", font=('Consolas', 10, 'bold'))
        self.log.tag_config("dim", foreground="#888888")

    def _choose_out(self):
        d = filedialog.askdirectory(initialdir=self.out_var.get())
        if d:
            self.out_var.set(d)

    def _log(self, msg, tag=None):
        try:
            self.log.config(state=tk.NORMAL)
            self.log.insert(tk.END, msg + "\n", tag or "")
            self.log.see(tk.END)
            self.log.config(state=tk.DISABLED)
        except Exception:
            pass

    def _log_from_thread(self, msg, tag=None):
        try:
            self.win.after(0, lambda: self._log(msg, tag))
        except Exception:
            pass

    def _set_prog(self, pct, text):
        def do():
            try:
                self.progress_var.set(pct)
                self.progress_text.set(text)
            except Exception:
                pass
        try:
            self.win.after(0, do)
        except Exception:
            pass

    def start(self):
        if self.running:
            return
        try:
            topk = int(self.topk_var.get())
            n_calls = int(self.calls_var.get())
            iou_thr = float(self.iou_var.get())
        except Exception as e:
            messagebox.showerror("参数错误", f"{e}")
            return

        out_dir = Path(self.out_var.get())
        if not out_dir.exists():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录: {e}")
                return

        reviewed = self.app.ground_truth.all_reviewed_items()
        if not reviewed:
            messagebox.showinfo("提示", "GT 库没有已审核的图")
            return

        self.running = True
        self.cancel_flag = False
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_use.config(state=tk.DISABLED)
        self.log.config(state=tk.NORMAL); self.log.delete("1.0", tk.END); self.log.config(state=tk.DISABLED)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"soup_{ts}.pt"

        threading.Thread(target=self._do_soup,
                         args=(topk, n_calls, iou_thr, out_path, reviewed),
                         daemon=True).start()

    def stop(self):
        self.cancel_flag = True
        self._log_from_thread("[停止] 用户请求停止", "warn")

    def _do_soup(self, topk, n_calls, iou_thr, out_path, reviewed):
        try:
            from predict import Detector
            from model_search import smart_soup, Candidate
            from model_soup import soup_models

            self._log_from_thread("=" * 60, "title")
            self._log_from_thread(f"候选数: {len(self.candidates)}  top_k: {topk}  "
                                  f"n_calls: {n_calls}  IoU: {iou_thr}", "title")
            self._log_from_thread("=" * 60, "title")

            self._log_from_thread("\n[步骤 1/3] 逐个评估候选模型...", "title")
            evaluated = []
            n_cand = len(self.candidates)
            for i, r in enumerate(self.candidates, 1):
                if self.cancel_flag:
                    self._log_from_thread("  已取消", "warn")
                    return
                self._set_prog((i - 1) / n_cand * 40, f"评估候选 {i}/{n_cand}")
                try:
                    detector = Detector(r["path"])
                except Exception as e:
                    self._log_from_thread(f"  ✗ {r['name']} 加载失败: {e}", "err")
                    continue
                total_f1 = 0.0
                n = 0
                for key, gt_rec in reviewed:
                    if self.cancel_flag:
                        return
                    img_path = self.app.ground_truth.img_dir / gt_rec["file"]
                    if not img_path.exists():
                        continue
                    try:
                        with Image.open(img_path) as im:
                            img_size = im.size
                    except Exception:
                        continue
                    try:
                        preds = detector.predict(str(img_path), conf=0.3)
                    except Exception:
                        continue
                    m = evaluate_one(gt_rec["boxes"], preds, iou_thr=iou_thr,
                                     img_size=img_size,
                                     use_area_weight=True,
                                     enforce_required=True)
                    total_f1 += m["f1"]
                    n += 1
                avg = total_f1 / n if n > 0 else 0.0
                cand = Candidate(path=r["path"], score=avg, class_scores={})
                evaluated.append(cand)
                try:
                    del detector
                except Exception:
                    pass
                self._log_from_thread(f"  {r['name']}  F1={avg:.4f}")

            if len(evaluated) < 2:
                self._log_from_thread("有效候选不足 2 个，终止", "err")
                return

            self._log_from_thread(f"\n[步骤 2/3] 搜索最优融合权重 ({n_calls} 次)...", "title")
            self._set_prog(45, "搜索融合权重...")

            def cb(msg):
                self._log_from_thread("  " + msg, "dim")

            paths, best_w, best_f1 = smart_soup(
                evaluated,
                [(k, {**dict(rec), "_img_dir": str(self.app.ground_truth.img_dir)})
                 for k, rec in reviewed],
                Detector,
                n_calls=n_calls,
                n_initial=min(10, n_calls),
                top_k=min(topk, len(evaluated)),
                verbose_cb=cb,
            )

            if self.cancel_flag:
                self._log_from_thread("已取消", "warn")
                return

            self._log_from_thread(f"\n[步骤 3/3] 输出融合模型...", "title")
            self._set_prog(85, "保存融合模型...")
            soup_models(paths, weights=best_w, output_path=str(out_path))
            self._log_from_thread(f"  ✓ 已输出: {out_path}", "ok")
            self._log_from_thread(f"  ✓ 最优 F1: {best_f1:.4f}", "ok")
            self._log_from_thread("  融合权重:", "title")
            for p, w in zip(paths, best_w):
                self._log_from_thread(f"    {Path(p).name}  w={w:.4f}", "dim")

            self.best_path = str(out_path)
            self._set_prog(100, f"完成 F1={best_f1:.4f}")
            self.win.after(0, lambda: self.btn_use.config(state=tk.NORMAL))
            self.win.after(0, lambda: messagebox.showinfo(
                "杂交完成",
                f"融合模型已保存:\n{out_path}\n\n"
                f"最优 GT-F1 = {best_f1:.4f}\n\n"
                f"可以点「✅ 用融合模型」立即切换。"))
        except Exception as e:
            import traceback as _tb
            self._log_from_thread(f"杂交异常: {e}", "err")
            self._log_from_thread(_tb.format_exc(), "err")
        finally:
            self.running = False
            self.win.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
            self.win.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))
            try:
                self.finder.win.after(500, self.finder.refresh)
            except Exception:
                pass

    def use_result(self):
        if not self.best_path or not Path(self.best_path).exists():
            messagebox.showinfo("提示", "还没有可用的融合模型")
            return
        if not messagebox.askyesno("切换模型",
                f"用这个融合模型？\n\n{self.best_path}"):
            return
        self.app.weights_pred.set(self.best_path)
        try:
            from predict import Detector
            self.app.detector = Detector(self.best_path)
            if hasattr(self.app, "gt_tab"):
                self.app.gt_tab.refresh_model_info()
            self.app.status.set(f"已切换到融合模型: {Path(self.best_path).name}")
            messagebox.showinfo("成功", "已切换")
        except Exception as e:
            messagebox.showerror("失败", f"{type(e).__name__}: {e}")

    def _on_close(self):
        if self.running:
            if not messagebox.askyesno("确认", "正在搜索，确定关闭？"):
                return
            self.cancel_flag = True
        try:
            self.win.destroy()
        except Exception:
            pass


def open_model_finder(parent, app):
    ModelFinderWindow(parent, app)