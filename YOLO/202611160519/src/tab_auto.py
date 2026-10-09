# tab_auto.py
# 自动标注页（v1.2）
# 新增：
#   - SAM 点击分割（可选）
#   - 拖拽 + GrabCut 精修
#   - Shift 多选 + 合并
#   - ESC 关闭子窗口
import copy
import os
import shutil
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO

from core import (CLASS_NAMES, CLASS_NAME_TO_ID, CLASS_COLORS,
                  IMG_EXTS, COCO_TO_PROJECT, parse_weak_label,
                  merge_boxes_union, merge_boxes_wbf,
                  suggest_duplicate_groups,
                  grabcut_rect, fastsam_rect, fastsam_model_exists)

SHIFT_MASK = 0x0001


class AutoTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.pending_queue = []
        self.pending_idx = 0
        self.pending_shown = 0

        self.current_image_path = None
        self.pending_review_image = None
        self.pending_review_photo = None
        self.pending_review_scale = 1.0
        self.pending_review_offset = (0, 0)
        self._np_rgb_cache = None

        self.pending_detections = []
        self.pending_selected_idx = None
        self.pending_selected_indices = []

        self.auto_label_mode = tk.BooleanVar(value=False)
        self.auto_manual_class = tk.StringVar(value="obstacle")
        self.auto_draw_mode = tk.StringVar(value="rect")   # rect / sam
        self.auto_use_grabcut = tk.BooleanVar(value=False)
        self.auto_drag_start = None
        self.auto_history = []

        self.review_limit = tk.IntVar(value=50)
        self.review_status = tk.StringVar(value="未开始审核")
        self.review_img_name = tk.StringVar(value="-")
        self.auto_conf = tk.DoubleVar(value=0.25)
        self.auto_conf_label_text = tk.StringVar(value="0.25")

        self.browse_list = []
        self.browse_index = -1

        self.queue_window = None
        self._build_ui()

    # ==================== UI ====================
    def _build_ui(self):
        app = self.app

        f = ttk.LabelFrame(self, text="自动标注参数")
        f.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(f, text="模型权重:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        self.auto_weight_combo = ttk.Combobox(f, textvariable=app.auto_weights,
                                              values=[], width=67, state="readonly")
        self.auto_weight_combo.grid(row=0, column=1, padx=5)
        ttk.Button(f, text="手动", command=self._choose_weights).grid(row=0, column=2, padx=5)

        for i, (lbl, var, cb) in enumerate([
            ("图片目录:", app.auto_img_dir, self._choose_img_dir),
            ("待审核标签目录:", app.auto_pending_dir, self._choose_pending_dir),
            ("正式标签目录:", app.auto_labels_dir, self._choose_labels_dir)], start=1):
            ttk.Label(f, text=lbl).grid(row=i, column=0, sticky=tk.W, padx=5, pady=4)
            ttk.Entry(f, textvariable=var, width=70).grid(row=i, column=1, padx=5)
            ttk.Button(f, text="选择", command=cb).grid(row=i, column=2, padx=5)

        r = ttk.Frame(f)
        r.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        ttk.Label(r, text="置信度:").pack(side=tk.LEFT)
        ttk.Scale(r, from_=0.05, to=0.95, variable=self.auto_conf,
                  orient=tk.HORIZONTAL, length=200,
                  command=lambda v: self.auto_conf_label_text.set(f"{float(v):.2f}")).pack(side=tk.LEFT, padx=5)
        ttk.Label(r, textvariable=self.auto_conf_label_text).pack(side=tk.LEFT)

        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=10, pady=4)
        self.btn_start_auto = ttk.Button(btns, text="① 开始自动标注", command=self.start_auto_label)
        self.btn_start_auto.pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="待审核列表", command=self.show_queue_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="清空待审核", command=self.clear_pending).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="打开待审核目录",
                   command=lambda: os.startfile(app.auto_pending_dir.get())
                   if Path(app.auto_pending_dir.get()).exists() else None).pack(side=tk.LEFT, padx=5)

        review = ttk.LabelFrame(self, text="② 人工审核 + 补标 + 合并")
        review.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        top2 = ttk.Frame(review)
        top2.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(top2, text="审核上限:").pack(side=tk.LEFT)
        ttk.Entry(top2, textvariable=self.review_limit, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(top2, text="张").pack(side=tk.LEFT)
        ttk.Button(top2, text="加载全部队列", command=self.load_queue).pack(side=tk.LEFT, padx=8)
        ttk.Button(top2, text="开始审核", command=self.start_review).pack(side=tk.LEFT, padx=4)
        ttk.Label(top2, textvariable=self.review_status, foreground="#007acc").pack(side=tk.RIGHT, padx=6)

        tools = ttk.Frame(review)
        tools.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Checkbutton(tools, text="补标模式", variable=self.auto_label_mode,
                        command=self._toggle_mode).pack(side=tk.LEFT, padx=4)
        ttk.Label(tools, text="画框:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Radiobutton(tools, text="拖拽", variable=self.auto_draw_mode,
                        value="rect").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(tools, text="SAM", variable=self.auto_draw_mode,
                        value="sam").pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(tools, text="GrabCut 精修",
                        variable=self.auto_use_grabcut).pack(side=tk.LEFT, padx=6)
        ttk.Label(tools, text="类别:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Combobox(tools, textvariable=self.auto_manual_class,
                     values=["obstacle", "cola", "football"], width=10,
                     state="readonly").pack(side=tk.LEFT, padx=2)

        tools2 = ttk.Frame(review)
        tools2.pack(fill=tk.X, padx=6, pady=(0, 4))
        for txt, cb in [("改类别", self.apply_class),
                        ("删框", self.delete_selected),
                        ("清空", self.clear_boxes),
                        ("↶ 撤销", self.undo),
                        ("🔗 合并选中框", self.merge_selected),
                        ("🔍 建议合并", self.auto_suggest_merge),
                        ("另存为训练数据", self.save_as_training),
                        ("存为标准答案", self.save_as_gt)]:
            ttk.Button(tools2, text=txt, command=cb).pack(side=tk.LEFT, padx=3)

        cf = ttk.Frame(review)
        cf.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas = tk.Canvas(cf, bg='#1e1e1e', highlightthickness=0, height=380)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda e: self._redraw())
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        ctl = ttk.Frame(review)
        ctl.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(ctl, text="⬅ 上一个", command=self.browse_prev).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctl, text="下一个 ➡", command=self.browse_next).pack(side=tk.LEFT, padx=3)
        ttk.Label(ctl, text="  |  ", foreground="#aaa").pack(side=tk.LEFT)
        ttk.Label(ctl, textvariable=self.review_img_name, font=('Consolas', 10)).pack(side=tk.LEFT)
        self.btn_accept = ttk.Button(ctl, text="✓ 通过", command=self.review_accept, state=tk.DISABLED)
        self.btn_accept.pack(side=tk.RIGHT, padx=3)
        self.btn_reject = ttk.Button(ctl, text="✗ 拒绝", command=self.review_reject, state=tk.DISABLED)
        self.btn_reject.pack(side=tk.RIGHT, padx=3)
        self.btn_skip = ttk.Button(ctl, text="跳过", command=self.review_skip, state=tk.DISABLED)
        self.btn_skip.pack(side=tk.RIGHT, padx=3)

    # ==================== 路径选择 ====================
    def _choose_weights(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if p: self.app.auto_weights.set(p)

    def _choose_img_dir(self):
        p = filedialog.askdirectory()
        if p: self.app.auto_img_dir.set(p)

    def _choose_pending_dir(self):
        p = filedialog.askdirectory()
        if p: self.app.auto_pending_dir.set(p)

    def _choose_labels_dir(self):
        p = filedialog.askdirectory()
        if p: self.app.auto_labels_dir.set(p)

    def clear_pending(self):
        d = Path(self.app.auto_pending_dir.get())
        if not d.exists(): return
        txts = list(d.glob("*.txt"))
        if not txts:
            messagebox.showinfo("提示", "已空"); return
        if messagebox.askyesno("确认", f"删除 {len(txts)} 个？"):
            for t in txts:
                try: t.unlink()
                except Exception: pass
            if self.queue_window and self.queue_window.winfo_exists():
                self.queue_window.destroy()
                self.queue_window = None

    # ==================== 自动标注 ====================
    def start_auto_label(self):
        app = self.app
        if not Path(app.auto_weights.get()).exists() or not Path(app.auto_img_dir.get()).exists():
            messagebox.showerror("错误", "权重或图片目录不存在"); return
        self.btn_start_auto.config(state=tk.DISABLED)
        app.status.set("自动标注中...")
        threading.Thread(target=self._do_auto, daemon=True).start()

    def _do_auto(self):
        try:
            app = self.app
            weights = app.auto_weights.get()
            img_dir = Path(app.auto_img_dir.get())
            out_dir = Path(app.auto_pending_dir.get())
            labels_dir = Path(app.auto_labels_dir.get())
            conf = float(self.auto_conf.get())

            out_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            done_stems = set()
            for p in labels_dir.glob("*.txt"): done_stems.add(p.stem)
            for p in out_dir.glob("*.txt"): done_stems.add(p.stem)

            model = YOLO(weights)
            is_custom = len(model.names) == 3
            imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

            total_boxes, skipped_exist, empty_count = 0, 0, 0
            for i, img_path in enumerate(imgs, 1):
                if img_path.stem in done_stems:
                    skipped_exist += 1
                    continue
                results = model.predict(img_path, conf=conf, verbose=False)
                lines = []
                for r in results:
                    ih, iw = r.orig_shape[0], r.orig_shape[1]
                    for box in r.boxes:
                        cid = int(box.cls)
                        if not is_custom:
                            cid = COCO_TO_PROJECT.get(cid, -1)
                            if cid == -1: continue
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx = ((x1 + x2) / 2) / iw
                        cy = ((y1 + y2) / 2) / ih
                        w = (x2 - x1) / iw
                        h = (y2 - y1) / ih
                        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                txt = out_dir / (img_path.stem + ".txt")
                with open(txt, "w", encoding="utf-8") as f:
                    if lines:
                        f.write("\n".join(lines) + "\n")
                        total_boxes += len(lines)
                    else:
                        f.write("")
                        empty_count += 1

            self.after(0, lambda: messagebox.showinfo(
                "自动标注完成",
                f"新增框数: {total_boxes}\n"
                f"跳过已标注: {skipped_exist}\n"
                f"无目标(空标签): {empty_count}\n\n"
                f"点「📋 待审核列表」逐张查看"))
        except Exception as e:
            print(traceback.format_exc())
            self.after(0, lambda: messagebox.showerror("出错", str(e)))
        finally:
            self.after(0, lambda: self.btn_start_auto.config(state=tk.NORMAL))
            self.after(0, lambda: self.app.status.set("就绪"))

    # ==================== 浏览上下文 ====================
    def _refresh_browse_list(self):
        pending_dir = Path(self.app.auto_pending_dir.get())
        if not pending_dir.exists():
            self.browse_list = []; return
        img_dirs = [Path(self.app.auto_img_dir.get())]
        parent = Path(self.app.auto_img_dir.get()).parent
        if (parent / "val").exists(): img_dirs.append(parent / "val")
        if (parent / "train").exists(): img_dirs.append(parent / "train")
        items = []
        for txt in sorted(pending_dir.glob("*.txt")):
            found = None
            for d in img_dirs:
                for ext in IMG_EXTS:
                    p = d / (txt.stem + ext)
                    if p.exists(): found = p; break
                if found: break
            if found: items.append((found, txt))
        self.browse_list = items

    def _get_browse_index(self):
        if not self.current_image_path:
            return -1
        cur = Path(self.current_image_path)
        for i, (img, _) in enumerate(self.browse_list):
            if img == cur:
                return i
        return -1

    def browse_next(self):
        self._refresh_browse_list()
        if not self.browse_list:
            messagebox.showinfo("提示", "没有待审核的图片"); return
        idx = self._get_browse_index()
        new_idx = 0 if idx < 0 else idx + 1
        if new_idx >= len(self.browse_list):
            messagebox.showinfo("提示", "已经是最后一张"); return
        img_path, txt_path = self.browse_list[new_idx]
        self._load_image_by_path(img_path, txt_path)
        self.app.status.set(f"待审核 {new_idx+1}/{len(self.browse_list)}: {img_path.name}")

    def browse_prev(self):
        self._refresh_browse_list()
        if not self.browse_list:
            messagebox.showinfo("提示", "没有待审核的图片"); return
        idx = self._get_browse_index()
        if idx <= 0:
            messagebox.showinfo("提示", "已经是第一张"); return
        img_path, txt_path = self.browse_list[idx - 1]
        self._load_image_by_path(img_path, txt_path)
        self.app.status.set(f"待审核 {idx}/{len(self.browse_list)}: {img_path.name}")

    # ==================== 待审核列表窗口 ====================
    def show_queue_window(self):
        app = self.app
        pending_dir = Path(app.auto_pending_dir.get())
        if not pending_dir.exists():
            messagebox.showinfo("提示", "待审核目录不存在，请先执行自动标注"); return

        img_dirs = [Path(app.auto_img_dir.get())]
        parent = Path(app.auto_img_dir.get()).parent
        if (parent / "val").exists(): img_dirs.append(parent / "val")
        if (parent / "train").exists(): img_dirs.append(parent / "train")

        items = []
        for txt in sorted(pending_dir.glob("*.txt")):
            found = None
            for d in img_dirs:
                for ext in IMG_EXTS:
                    p = d / (txt.stem + ext)
                    if p.exists(): found = p; break
                if found: break
            boxes = []
            try:
                with open(txt, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            boxes.append(int(parts[0]))
            except Exception:
                pass
            cls_cnt = {}
            for c in boxes:
                name = CLASS_NAMES.get(c, str(c))
                cls_cnt[name] = cls_cnt.get(name, 0) + 1
            items.append({"txt": txt, "img": found, "boxes": boxes,
                          "cls_cnt": cls_cnt, "stem": txt.stem})

        if not items:
            messagebox.showinfo("提示", "待审核目录为空"); return

        if self.queue_window is not None and self.queue_window.winfo_exists():
            self.queue_window.deiconify()
            self.queue_window.lift()
            for child in self.queue_window.winfo_children():
                child.destroy()
        else:
            self.queue_window = tk.Toplevel(self)
            self.queue_window.geometry("1000x680")
        win = self.queue_window
        win.title(f"待审核列表 - {len(items)} 张  (支持多选)")
        win.protocol("WM_DELETE_WINDOW", lambda: win.withdraw())
        win.lift()
        win.focus()
        win.bind("<Escape>", lambda e: win.withdraw())

        total_boxes = sum(len(it["boxes"]) for it in items)
        with_img = sum(1 for it in items if it["img"] is not None)
        empty_lbl = sum(1 for it in items if len(it["boxes"]) == 0)

        stat = ttk.LabelFrame(win, text="汇总")
        stat.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(stat,
                  text=f"共 {len(items)} 张  |  有对应图片 {with_img} 张  |  "
                       f"总框数 {total_boxes}  |  空标签 {empty_lbl} 张",
                  font=('Consolas', 11), foreground="#007acc").pack(anchor=tk.W, padx=8, pady=6)

        search_row = ttk.Frame(win)
        search_row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(search_row, text="筛选:").pack(side=tk.LEFT)
        filter_var = tk.StringVar()
        ttk.Entry(search_row, textvariable=filter_var, width=40).pack(side=tk.LEFT, padx=6)
        ttk.Label(search_row, text="(文件名关键字)", foreground="#666").pack(side=tk.LEFT)

        listf = ttk.LabelFrame(win, text="逐张结果（双击审核，Ctrl+A 全选，Shift+点击连选）")
        listf.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        cols = ("idx", "image", "boxes", "cola", "football", "obstacle")
        tree = ttk.Treeview(listf, columns=cols, show="headings", height=20,
                            selectmode="extended")
        for c, t, w in [("idx", "#", 50), ("image", "图片", 420), ("boxes", "总框", 60),
                        ("cola", "cola", 60), ("football", "football", 70), ("obstacle", "obstacle", 70)]:
            tree.heading(c, text=t)
            tree.column(c, width=w, anchor="center" if c != "image" else "w")
        sb = ttk.Scrollbar(listf, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        iid_map = {}

        def refill():
            for iid in tree.get_children():
                tree.delete(iid)
            iid_map.clear()
            kw = filter_var.get().strip().lower()
            for i, it in enumerate(items):
                if kw and kw not in it["stem"].lower():
                    continue
                if it["img"] is None:
                    img_display = f"{it['stem']} (无对应图片)"
                    tags = ("missing",)
                else:
                    img_display = it["img"].name
                    tags = ()
                iid = str(i)
                tree.insert("", "end", iid=iid,
                            values=(i + 1, img_display, len(it["boxes"]),
                                    it["cls_cnt"].get("cola", 0),
                                    it["cls_cnt"].get("football", 0),
                                    it["cls_cnt"].get("obstacle", 0)),
                            tags=tags)
                iid_map[iid] = it
        tree.tag_configure("missing", foreground="#cc0000")
        refill()
        filter_var.trace_add('write', lambda *a: refill())

        def on_double(event):
            sel = tree.selection()
            if not sel: return
            it = iid_map.get(sel[0])
            if it is None: return
            if it["img"] is None:
                messagebox.showerror("错误", f"找不到对应图片:\n{it['stem']}", parent=win); return
            win.withdraw()
            self._load_image_by_path(it["img"], it["txt"])
            self.app.status.set(f"请审核: {it['img'].name}  (用 ⬅/➡ 切换)")
        tree.bind("<Double-Button-1>", on_double)

        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(btns, text="双击行 → 跳回审核；支持多选批量操作",
                  foreground="#666").pack(side=tk.LEFT, padx=4)

        def select_all():
            tree.selection_set(tree.get_children())

        def invert_sel():
            selected = set(tree.selection())
            all_ids = set(tree.get_children())
            tree.selection_remove(*all_ids)
            for iid in (all_ids - selected):
                tree.selection_add(iid)

        def batch_load():
            sels = tree.selection()
            if not sels:
                messagebox.showinfo("提示", "请先选择要加载的项", parent=win); return
            selected_items = []
            for iid in sels:
                it = iid_map.get(iid)
                if it is None: continue
                if it["img"] is None: continue
                selected_items.append((it["img"], it["txt"]))
            if not selected_items:
                messagebox.showinfo("提示", "选中的项没有对应图片", parent=win); return
            self.pending_queue = selected_items
            self.pending_idx = 0
            self.pending_shown = 0
            self.review_limit.set(max(len(selected_items), 1))
            win.withdraw()
            self._load_image(*selected_items[0])
            self.review_img_name.set(f"{selected_items[0][0].name}  (1/{len(selected_items)})")
            self.btn_accept.config(state=tk.NORMAL)
            self.btn_reject.config(state=tk.NORMAL)
            self.btn_skip.config(state=tk.NORMAL)
            self.app.status.set(f"队列审核: 共 {len(selected_items)} 张")

        def batch_delete():
            sels = tree.selection()
            if not sels:
                messagebox.showinfo("提示", "请先选择要删除的项", parent=win); return
            if not messagebox.askyesno("确认",
                    f"删除 {len(sels)} 个待审核文件？\n（只删除待审核目录里的 .txt，不影响图片）",
                    parent=win):
                return
            deleted = 0
            for iid in sels:
                it = iid_map.get(iid)
                if it is None: continue
                try:
                    it["txt"].unlink()
                    deleted += 1
                except Exception:
                    pass
            messagebox.showinfo("完成", f"已删除 {deleted} 个", parent=win)
            win.destroy()
            self.queue_window = None
            self.show_queue_window()

        ttk.Button(btns, text="全选", command=select_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="反选", command=invert_sel).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="批量加载", command=batch_load).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="批量删除", command=batch_delete).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="刷新",
                   command=lambda: [win.destroy(), setattr(self, "queue_window", None),
                                    self.show_queue_window()]).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="隐藏", command=win.withdraw).pack(side=tk.RIGHT, padx=4)

    # ==================== 单张加载 ====================
    def _load_image_by_path(self, img_path, txt_path):
        try:
            self.pending_queue = [(Path(img_path), Path(txt_path))]
            self.pending_idx = 0
            self.pending_shown = 0
            self.review_img_name.set(f"{Path(img_path).name}  (单张)")
            self.review_status.set("单张审核")
            self._load_image(Path(img_path), Path(txt_path))
            self.btn_accept.config(state=tk.NORMAL)
            self.btn_reject.config(state=tk.NORMAL)
            self.btn_skip.config(state=tk.NORMAL)
            self.auto_history.clear()
            self._refresh_browse_list()
        except Exception as e:
            messagebox.showerror("加载失败", f"{type(e).__name__}: {e}")

    # ==================== 队列审核 ====================
    def load_queue(self):
        app = self.app
        pending_dir = Path(app.auto_pending_dir.get())
        if not pending_dir.exists():
            messagebox.showinfo("提示", "待审核目录不存在"); return
        img_dirs = [Path(app.auto_img_dir.get())]
        parent = Path(app.auto_img_dir.get()).parent
        if (parent / "val").exists(): img_dirs.append(parent / "val")
        q = []
        for txt in sorted(pending_dir.glob("*.txt")):
            found = None
            for d in img_dirs:
                for ext in IMG_EXTS:
                    p = d / (txt.stem + ext)
                    if p.exists(): found = p; break
                if found: break
            if found: q.append((found, txt))
        self.pending_queue = q
        self.pending_idx = 0
        self.pending_shown = 0
        self.review_status.set(f"队列: {len(q)} 张")
        self.btn_accept.config(state=tk.DISABLED)
        self.btn_reject.config(state=tk.DISABLED)
        self.btn_skip.config(state=tk.DISABLED)
        self.canvas.delete("all")
        self.pending_detections = []
        self.pending_selected_idx = None
        self.pending_selected_indices = []
        self.auto_history.clear()
        messagebox.showinfo("已加载", f"待审核队列: {len(q)} 张\n点「开始审核」开始")

    def start_review(self):
        if not self.pending_queue:
            messagebox.showinfo("提示", "请先加载队列"); return
        self.pending_idx = 0
        self.pending_shown = 0
        self.show_next()

    def show_next(self):
        if self.pending_idx >= len(self.pending_queue):
            self._end_review("队列处理完毕"); return
        if self.pending_shown >= int(self.review_limit.get()):
            self._end_review(f"达到上限 {self.review_limit.get()}"); return
        img_path, txt_path = self.pending_queue[self.pending_idx]
        self.review_img_name.set(
            f"{img_path.name}  ({self.pending_shown+1}/{self.review_limit.get()})")
        self.review_status.set(f"进度: {self.pending_idx+1}/{len(self.pending_queue)}")
        self._load_image(img_path, txt_path)
        self.btn_accept.config(state=tk.NORMAL)
        self.btn_reject.config(state=tk.NORMAL)
        self.btn_skip.config(state=tk.NORMAL)
        self.auto_history.clear()

    def _end_review(self, msg):
        self.btn_accept.config(state=tk.DISABLED)
        self.btn_reject.config(state=tk.DISABLED)
        self.btn_skip.config(state=tk.DISABLED)
        self.review_status.set(msg)
        messagebox.showinfo("完成", msg)

    # ==================== 图像加载与渲染 ====================
    def _load_image(self, img_path, txt_path):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.review_status.set(f"无法打开: {e}"); return
        iw, ih = img.size
        self.current_image_path = str(img_path)
        self._np_rgb_cache = None
        self.pending_detections = []
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = (cx - w / 2) * iw
                    y1 = (cy - h / 2) * ih
                    x2 = (cx + w / 2) * iw
                    y2 = (cy + h / 2) * ih
                    self.pending_detections.append({
                        "xyxy": [x1, y1, x2, y2],
                        "cls_id": cid,
                        "cls_name": CLASS_NAMES.get(cid, str(cid)),
                        "conf": 1.0,
                        "manual": False,
                    })
        except Exception:
            pass
        self.pending_review_image = img
        self.pending_selected_idx = None
        self.pending_selected_indices = []
        self._redraw()

    def _redraw(self):
        if self.pending_review_image is None: return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            self.after(100, self._redraw); return
        img = self.pending_review_image.copy()
        draw = ImageDraw.Draw(img)
        font = self._get_font(16)

        sel_set = set(self.pending_selected_indices)

        for i, d in enumerate(self.pending_detections):
            x1, y1, x2, y2 = d["xyxy"]
            if i in sel_set:
                color, width = '#FFFF00', 4
            elif d.get("manual"):
                color, width = '#00FFFF', 3
            else:
                color, width = CLASS_COLORS[d["cls_id"] % len(CLASS_COLORS)], 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label = d['cls_name'] + (" [手动]" if d.get("manual") else "")
            if i in sel_set: label += " [选]"
            try:
                bbox = draw.textbbox((x1, y1), label, font=font)
            except Exception:
                bbox = (x1, y1, x1 + len(label) * 9, y1 + 18)
            ty = max(0, bbox[1] - 2)
            draw.rectangle([bbox[0] - 2, ty, bbox[2] + 2, bbox[3] + 2], fill=color)
            draw.text((x1, ty), label, fill='black', font=font)

        if not self.pending_detections:
            tip = "⚠ 此图无检测框（自动标注为空，可能是真无目标或模型没检出）"
            try:
                bbox = draw.textbbox((0, 0), tip, font=font)
            except Exception:
                bbox = (0, 0, 500, 20)
            tw = bbox[2] - bbox[0]
            draw.rectangle([5, 5, 5 + tw + 20, 45], fill='#cc0000')
            draw.text((15, 15), tip, fill='white', font=font)
        else:
            n_sel = len(self.pending_selected_indices)
            tip = (f"已选 {n_sel} 个框  |  Shift+点击多选，点「🔗 合并选中框」"
                   if n_sel >= 2 else "Shift+点击可多选")
            try:
                bbox = draw.textbbox((0, 0), tip, font=font)
            except Exception:
                bbox = (0, 0, 400, 24)
            tw = bbox[2] - bbox[0]
            draw.rectangle([5, 5, 5 + tw + 20, 38], fill='#0088cc')
            draw.text((13, 12), tip, fill='white', font=font)

        iw, ih = img.size
        self.pending_review_scale = min(cw / iw, ch / ih)
        nw = max(1, int(iw * self.pending_review_scale))
        nh = max(1, int(ih * self.pending_review_scale))
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.pending_review_offset = (ox, oy)
        resized = img.resize((nw, nh), Image.LANCZOS)
        self.pending_review_photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.pending_review_photo)

    # ==================== 补标 + 多选 + SAM ====================
    def _toggle_mode(self):
        self.canvas.config(cursor="cross" if self.auto_label_mode.get() else "")

    def _hit_test(self, ix, iy):
        hit = None
        for i in range(len(self.pending_detections) - 1, -1, -1):
            x1, y1, x2, y2 = self.pending_detections[i]["xyxy"]
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                hit = i
                break
        return hit

    def _on_press(self, event):
        if self.pending_review_image is None: return
        if self.auto_label_mode.get():
            ix = (event.x - self.pending_review_offset[0]) / self.pending_review_scale
            iy = (event.y - self.pending_review_offset[1]) / self.pending_review_scale
            iw, ih = self.pending_review_image.size
            if not (0 <= ix < iw and 0 <= iy < ih):
                return
            if self.auto_draw_mode.get() == "sam":
                self._do_sam_click(ix, iy)
                return
            self.auto_drag_start = (ix, iy); return
        if not self.pending_detections: return
        ix = (event.x - self.pending_review_offset[0]) / self.pending_review_scale
        iy = (event.y - self.pending_review_offset[1]) / self.pending_review_scale
        hit = self._hit_test(ix, iy)

        is_shift = bool(event.state & SHIFT_MASK)
        if hit is None:
            if not is_shift:
                self.pending_selected_indices = []
                self.pending_selected_idx = None
        else:
            if is_shift:
                if hit in self.pending_selected_indices:
                    self.pending_selected_indices.remove(hit)
                else:
                    self.pending_selected_indices.append(hit)
            else:
                self.pending_selected_indices = [hit]
            self.pending_selected_idx = hit
        self._redraw()

    def _do_sam_click(self, ix, iy):
        try:
            img = self._np_rgb_cache
            if img is None:
                img = np.array(self.pending_review_image)
                self._np_rgb_cache = img

            if not fastsam_model_exists("FastSAM-s.pt"):
                if not messagebox.askyesno(
                    "首次使用 SAM",
                    "SAM 点击分割需要下载模型文件 FastSAM-s.pt（约 23MB）。\n\n"
                    "是否现在下载？\n"
                    "（选「否」将改用 GrabCut，需要先拖一个大概的框）",
                    icon="question"):
                    self.app.status.set("已跳过 SAM，使用拖拽 + GrabCut")
                    return
                self.app.status.set("正在下载 FastSAM-s.pt，请稍候...")
                self.update_idletasks()

            self.app.status.set("SAM 分割中...")
            self.update_idletasks()
            rect = fastsam_rect(img, int(ix), int(iy))
            if rect is None:
                messagebox.showinfo("SAM 未圈到", "SAM 没圈出物体，可以改用拖拽 + GrabCut")
                return

            x1, y1, x2, y2 = rect
            self._push()
            cls_name = self.auto_manual_class.get()
            cid = CLASS_NAME_TO_ID[cls_name]
            self.pending_detections.append({
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": 1.0, "cls_id": cid, "cls_name": cls_name, "manual": True})
            self._redraw()
            self.app.status.set(f"SAM 添加: {cls_name}")
        except Exception as e:
            print(traceback.format_exc())
            self.app.status.set(f"SAM 失败: {e}")
            messagebox.showerror("SAM 失败", f"{type(e).__name__}: {e}")

    def _on_drag(self, event):
        if not self.auto_label_mode.get() or self.auto_drag_start is None: return
        if self.auto_draw_mode.get() == "sam": return
        ix = (event.x - self.pending_review_offset[0]) / self.pending_review_scale
        iy = (event.y - self.pending_review_offset[1]) / self.pending_review_scale
        x1, y1 = self.auto_drag_start
        cx1 = self.pending_review_offset[0] + x1 * self.pending_review_scale
        cy1 = self.pending_review_offset[1] + y1 * self.pending_review_scale
        cx2 = self.pending_review_offset[0] + ix * self.pending_review_scale
        cy2 = self.pending_review_offset[1] + iy * self.pending_review_scale
        self.canvas.delete("drag_rect")
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline='#00ff00', width=3, tags="drag_rect")

    def _on_release(self, event):
        if not self.auto_label_mode.get() or self.auto_drag_start is None: return
        if self.auto_draw_mode.get() == "sam": return
        ix = (event.x - self.pending_review_offset[0]) / self.pending_review_scale
        iy = (event.y - self.pending_review_offset[1]) / self.pending_review_scale
        x1, y1 = self.auto_drag_start
        x2, y2 = ix, iy
        self.auto_drag_start = None
        self.canvas.delete("drag_rect")
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5: return
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        iw, ih = self.pending_review_image.size
        x1 = max(0, min(x1, iw)); x2 = max(0, min(x2, iw))
        y1 = max(0, min(y1, ih)); y2 = max(0, min(y2, ih))

        # GrabCut 精修
        if self.auto_use_grabcut.get() and self._np_rgb_cache is not None:
            try:
                self.app.status.set("GrabCut 精修中...")
                self.update_idletasks()
                x1, y1, x2, y2 = grabcut_rect(self._np_rgb_cache, (x1, y1, x2, y2))
            except Exception as e:
                print(f"[GrabCut 失败] {e}")

        self._push()
        cls_name = self.auto_manual_class.get()
        cid = CLASS_NAME_TO_ID[cls_name]
        self.pending_detections.append({
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "conf": 1.0, "cls_id": cid, "cls_name": cls_name, "manual": True})
        self._redraw()

    def _push(self):
        self.auto_history.append(copy.deepcopy(self.pending_detections))
        if len(self.auto_history) > 50: self.auto_history.pop(0)

    def undo(self):
        if not self.auto_history: return
        self.pending_detections = self.auto_history.pop()
        self.pending_selected_idx = None
        self.pending_selected_indices = []
        self._redraw()

    def apply_class(self):
        if not self.pending_selected_indices:
            messagebox.showinfo("提示", "请先选框"); return
        self._push()
        name = self.auto_manual_class.get()
        cid = CLASS_NAME_TO_ID[name]
        for i in self.pending_selected_indices:
            if 0 <= i < len(self.pending_detections):
                self.pending_detections[i]["cls_id"] = cid
                self.pending_detections[i]["cls_name"] = name
        self._redraw()

    def delete_selected(self):
        if not self.pending_selected_indices:
            messagebox.showinfo("提示", "请先选框"); return
        n = len(self.pending_selected_indices)
        if not messagebox.askyesno("确认", f"删除选中的 {n} 个框？"):
            return
        self._push()
        for i in sorted(self.pending_selected_indices, reverse=True):
            if 0 <= i < len(self.pending_detections):
                del self.pending_detections[i]
        self.pending_selected_idx = None
        self.pending_selected_indices = []
        self._redraw()

    def clear_boxes(self):
        if not self.pending_detections: return
        if messagebox.askyesno("确认", f"清空 {len(self.pending_detections)} 个框？"):
            self._push()
            self.pending_detections = []
            self.pending_selected_idx = None
            self.pending_selected_indices = []
            self._redraw()

    # ==================== 合并 ====================
    def merge_selected(self):
        if len(self.pending_selected_indices) < 2:
            messagebox.showinfo("提示", "请按住 Shift 多选至少两个框")
            return
        indices = sorted(self.pending_selected_indices)
        boxes = [self.pending_detections[i] for i in indices
                 if 0 <= i < len(self.pending_detections)]
        if len(boxes) < 2:
            return
        cls_ids = set(b["cls_id"] for b in boxes)
        if len(cls_ids) > 1:
            if not messagebox.askyesno("类别不同",
                    "选中的框类别不同，确定合并吗？\n合并后类别取置信度最高的那个。"):
                return
            method = "wbf"
            merged = merge_boxes_wbf(boxes)
        else:
            answer = messagebox.askyesnocancel(
                "合并方式",
                "「是」= 加权融合(WBF)（推荐）\n"
                "「否」= 并集外接矩形\n"
                "「取消」= 放弃合并")
            if answer is None:
                return
            method = "wbf" if answer else "union"
            merged = merge_boxes_wbf(boxes) if answer else merge_boxes_union(boxes)
        if merged is None:
            return
        self._push()
        for i in sorted(indices, reverse=True):
            if 0 <= i < len(self.pending_detections):
                del self.pending_detections[i]
        self.pending_detections.append(merged)
        self.pending_selected_idx = None
        self.pending_selected_indices = []
        if hasattr(self.app, "merge_log") and self.current_image_path:
            try:
                self.app.merge_log.record(
                    Path(self.current_image_path).name, len(boxes), 1, method)
            except Exception as e:
                print(f"[合并记录失败] {e}")
        self._redraw()
        self.app.status.set(f"已合并 {len(boxes)} 个框 ({method})")

    def auto_suggest_merge(self):
        if len(self.pending_detections) < 2:
            messagebox.showinfo("提示", "当前框不足 2 个"); return
        groups = suggest_duplicate_groups(self.pending_detections,
                                          iou_thr=0.55, contain_thr=0.80)
        if not groups:
            messagebox.showinfo("提示", "没发现建议合并的重复框"); return
        all_idx = []
        for g in groups:
            all_idx.extend(g)
        self.pending_selected_indices = sorted(set(all_idx))
        self.pending_selected_idx = self.pending_selected_indices[0] \
            if self.pending_selected_indices else None
        self._redraw()
        messagebox.showinfo(
            "建议合并",
            f"发现 {len(groups)} 组重复框，已全部选中。\n"
            f"共 {len(self.pending_selected_indices)} 个框。\n\n"
            f"点「🔗 合并选中框」逐组合并。")

    # ==================== 字体 ====================
    def _get_font(self, size):
        candidates = [
            "msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "arial.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _write_to(self, dst_lbl):
        iw, ih = self.pending_review_image.size
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for d in self.pending_detections:
                x1, y1, x2, y2 = d["xyxy"]
                cx = (x1 + x2) / 2 / iw
                cy = (y1 + y2) / 2 / ih
                w = (x2 - x1) / iw
                h = (y2 - y1) / ih
                f.write(f"{d['cls_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def save_as_training(self):
        if self.pending_review_image is None or not self.pending_queue:
            messagebox.showinfo("提示", "请先加载图片"); return
        img_path, _ = self.pending_queue[self.pending_idx]
        lbl_dir = Path(self.app.auto_labels_dir.get())
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_dir = Path(self.app.auto_img_dir.get())
        img_dir.mkdir(parents=True, exist_ok=True)
        dst_img = img_dir / img_path.name
        try:
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
        except Exception:
            pass
        self._write_to(lbl_dir / (img_path.stem + ".txt"))
        try:
            self.pending_queue[self.pending_idx][1].unlink()
        except Exception:
            pass
        self.app.status.set(f"已保存到训练集: {img_path.name}")
        self._auto_next_after_action()

    def save_as_gt(self):
        if self.pending_review_image is None or not self.pending_queue:
            messagebox.showinfo("提示", "请先加载图片"); return
        img_path, _ = self.pending_queue[self.pending_idx]
        try:
            self.app.ground_truth.add(img_path, self.pending_detections, reviewed=True)
            if hasattr(self.app, "review_tab"):
                self.app.review_tab.refresh_gt_list()
            if hasattr(self.app, "gt_tab"):
                self.app.gt_tab.refresh_gt_list()
            messagebox.showinfo("已保存",
                f"共 {len(self.pending_detections)} 个框已加入标准答案库（标记为已审核）")
        except Exception as e:
            messagebox.showerror("失败", f"{e}")

    def review_accept(self):
        img_path, txt_path = self.pending_queue[self.pending_idx]
        dst_dir = Path(self.app.auto_labels_dir.get())
        dst_dir.mkdir(parents=True, exist_ok=True)
        self._write_to(dst_dir / txt_path.name)
        try:
            txt_path.unlink()
        except Exception:
            pass
        self.app.status.set(f"✓ 通过: {img_path.name}")
        self._auto_next_after_action()

    def review_reject(self):
        img_path, txt_path = self.pending_queue[self.pending_idx]
        self.app.error_book.record(img_path.name, parse_weak_label(img_path.name),
                                   "manual_reject", [])
        try:
            txt_path.unlink()
        except Exception:
            pass
        if hasattr(self.app, "error_tab"):
            self.app.error_tab.refresh()
        self.app.status.set(f"✗ 拒绝: {img_path.name}")
        self._auto_next_after_action()

    def review_skip(self):
        self.pending_idx += 1
        self.pending_shown += 1
        self.show_next()

    def _auto_next_after_action(self):
        if len(self.pending_queue) > 1:
            self.pending_idx += 1
            self.pending_shown += 1
            self.show_next()
            return

        self._refresh_browse_list()
        if not self.browse_list:
            self.btn_accept.config(state=tk.DISABLED)
            self.btn_reject.config(state=tk.DISABLED)
            self.btn_skip.config(state=tk.DISABLED)
            self.review_img_name.set("-")
            self.review_status.set("待审核目录已清空")
            messagebox.showinfo("提示", "所有待审核图已处理完毕")
            return
        img_path, txt_path = self.browse_list[0]
        self._load_image_by_path(img_path, txt_path)
        self.app.status.set(f"下一张: {img_path.name}")