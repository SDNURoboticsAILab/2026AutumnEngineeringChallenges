# tab_error.py
# 错题本（v1.2）
# 修复：reopen_selected / batch_reopen 改调 gt_tab._load_gt_for_review
import os
import shutil
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk

from core import IMG_EXTS, STATUS_TEXT


class ErrorTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.stat_var = tk.StringVar(value="错题数: 0")
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(top, text="错题本（按错误次数降序）", font=('Arial', 11, 'bold')).pack(side=tk.LEFT)
        ttk.Button(top, text="刷新", command=self.refresh).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="打开记录",
                   command=lambda: os.startfile(str(self.app.error_book.path))
                   if self.app.error_book.path.exists() else None).pack(side=tk.RIGHT, padx=4)

        ctl = ttk.Frame(self); ctl.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(ctl, text="批量操作:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctl, text="全选", command=self.select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctl, text="反选", command=self.invert_selection).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctl, text="🔁 批量重审", command=self.batch_reopen).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctl, text="批量移除", command=self.batch_remove).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctl, text="批量加入训练集", command=self.batch_add_to_training).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctl, text="(Ctrl+A 全选, Shift+点击 连选)",
                  foreground="#666").pack(side=tk.RIGHT, padx=6)

        frame = ttk.Frame(self); frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        sb = ttk.Scrollbar(frame); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(frame, yscrollcommand=sb.set,
                                   font=('Consolas', 10), activestyle='none',
                                   selectmode=tk.EXTENDED)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.listbox.yview)
        self.listbox.bind('<Double-Button-1>', lambda e: self.reopen_selected())
        self.listbox.bind('<Control-a>', lambda e: (self.select_all(), "break"))

        ttk.Label(self, textvariable=self.stat_var, font=('Arial', 10)).pack(anchor=tk.W, padx=8, pady=4)

    def refresh(self):
        try:
            self.listbox.delete(0, tk.END)
            q = self.app.error_book.queue()
            for name, info in q:
                wl = info.get("weak_label") or "-"
                st = STATUS_TEXT.get(info.get("status", ""), info.get("status", ""))
                self.listbox.insert(tk.END,
                    f"×{info.get('wrong_count',0):<3}  {name:<40}  弱:{wl:<10}  {st}")
            self.stat_var.set(f"错题数: {len(q)}")
        except Exception as e:
            print(f"[错题本刷新失败] {e}")

    # ========== 多选辅助 ==========
    def select_all(self):
        self.listbox.selection_set(0, tk.END)

    def invert_selection(self):
        n = self.listbox.size()
        selected = set(self.listbox.curselection())
        self.listbox.selection_clear(0, tk.END)
        for i in range(n):
            if i not in selected:
                self.listbox.selection_set(i)

    # ========== 单个操作 ==========
    def remove_selected(self):
        sel = self.listbox.curselection()
        if not sel: messagebox.showinfo("提示", "请先选一条"); return
        q = self.app.error_book.queue()
        if sel[0] >= len(q): return
        name = q[sel[0]][0]
        if messagebox.askyesno("确认", f"移除?\n{name}"):
            self.app.error_book.remove(name); self.refresh()

    def reopen_selected(self):
        sel = self.listbox.curselection()
        if not sel: messagebox.showinfo("提示", "请先选一条"); return
        q = self.app.error_book.queue()
        if sel[0] >= len(q): return
        name = q[sel[0]][0]
        img_path = self._find_image(name)
        if img_path is None:
            messagebox.showerror("错误", f"找不到: {name}"); return
        if self.app.detector is None:
            if Path(self.app.weights_pred.get()).exists():
                try:
                    from predict import Detector
                    self.app.detector = Detector(self.app.weights_pred.get())
                except Exception as e:
                    messagebox.showerror("失败", f"{e}"); return
            else:
                messagebox.showerror("错误", "请先加载模型"); return
        self.app.notebook.select(self.app.gt_tab)
        # ===== 修复：改用 gt_tab._load_gt_for_review =====
        rec = self.app.ground_truth.get_by_key(name)
        if rec:
            gt_img = self.app.ground_truth.img_dir / rec["file"]
            if gt_img.exists():
                self.app.gt_tab._load_gt_for_review(name, gt_img, rec["boxes"])
            else:
                self.app.gt_tab.load_image_path(str(img_path))
        else:
            # 不在 GT 库里，直接用图片打开
            if hasattr(self.app.gt_tab, "load_image_path"):
                self.app.gt_tab.load_image_path(str(img_path))
            else:
                self.app.gt_tab.current_image_path = str(img_path)
                from PIL import Image
                self.app.gt_tab.current_image = Image.open(img_path).convert("RGB")
                self.app.gt_tab.detections = []
                self.app.gt_tab.selected_indices = []
                self.app.gt_tab._build_edge_cache()
                self.app.gt_tab.update_canvas()
                self.app.gt_tab.render()

    # ========== 批量操作 ==========
    def batch_remove(self):
        sel = self.listbox.curselection()
        if not sel: messagebox.showinfo("提示", "请先选择"); return
        q = self.app.error_book.queue()
        names = [q[i][0] for i in sel if i < len(q)]
        if not names: return
        if messagebox.askyesno("确认", f"移除 {len(names)} 条？"):
            for name in names:
                self.app.error_book.remove(name)
            self.refresh()

    def batch_reopen(self):
        sel = self.listbox.curselection()
        if not sel: messagebox.showinfo("提示", "请先选择"); return
        q = self.app.error_book.queue()
        names = [q[i][0] for i in sel if i < len(q)]
        first = None
        for name in names:
            p = self._find_image(name)
            if p is not None: first = (name, p); break
        if first is None:
            messagebox.showerror("错误", "选中的图都找不到"); return
        name, img_path = first
        if self.app.detector is None:
            if Path(self.app.weights_pred.get()).exists():
                try:
                    from predict import Detector
                    self.app.detector = Detector(self.app.weights_pred.get())
                except Exception as e:
                    messagebox.showerror("失败", f"{e}"); return
            else:
                messagebox.showerror("错误", "请先加载模型"); return
        self.app.notebook.select(self.app.gt_tab)
        rec = self.app.ground_truth.get_by_key(name)
        if rec:
            gt_img = self.app.ground_truth.img_dir / rec["file"]
            if gt_img.exists():
                self.app.gt_tab._load_gt_for_review(name, gt_img, rec["boxes"])
        else:
            if hasattr(self.app.gt_tab, "load_image_path"):
                self.app.gt_tab.load_image_path(str(img_path))
        messagebox.showinfo("提示",
            f"已加载第 1 张（共 {len(names)} 张选中）\n\n"
            f"逐张审核后返回错题本继续下一张")

    def batch_add_to_training(self):
        sel = self.listbox.curselection()
        if not sel: messagebox.showinfo("提示", "请先选择"); return
        q = self.app.error_book.queue()
        names = [q[i][0] for i in sel if i < len(q)]
        if not names: return

        img_dir = Path(self.app.auto_img_dir.get()); lbl_dir = Path(self.app.auto_labels_dir.get())
        img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)

        added = 0; skipped = 0
        for name in names:
            rec = self.app.ground_truth.get_by_key(name)
            if not rec:
                skipped += 1; continue
            src = self.app.ground_truth.img_dir / rec["file"]
            if not src.exists():
                skipped += 1; continue
            try:
                from PIL import Image as _Img
                iw, ih = _Img.open(src).size
                shutil.copy2(src, img_dir / src.name)
                with open(lbl_dir / (src.stem + ".txt"), "w", encoding="utf-8") as f:
                    for b in rec["boxes"]:
                        x1, y1, x2, y2 = b["xyxy"]
                        cx = (x1+x2)/2/iw; cy = (y1+y2)/2/ih
                        w = (x2-x1)/iw; h = (y2-y1)/ih
                        f.write(f"{b['cls_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                added += 1
            except Exception as e:
                print(f"[加入训练集失败] {name}: {e}")
                skipped += 1

        messagebox.showinfo("完成",
            f"成功加入训练集: {added} 张\n跳过: {skipped} 张\n\n"
            f"注意：加入的是 GT 里的框，不是模型检测的框")

    def _find_image(self, name):
        rec = self.app.ground_truth.get_by_key(name)
        if rec:
            p = self.app.ground_truth.img_dir / rec["file"]
            if p.exists(): return p
        stem = Path(name).stem
        for d in [Path(self.app.auto_img_dir.get()),
                  Path(self.app.auto_img_dir.get()).parent / "val",
                  Path(self.app.auto_img_dir.get()).parent / "train"]:
            for ext in IMG_EXTS:
                p = d / (stem + ext)
                if p.exists(): return p
        return None