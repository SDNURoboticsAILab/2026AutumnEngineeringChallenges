# tab_gt.py
# GT 库管理页（v1.3）
# 新增：
#   - SAM 点击分割（可选，首次弹下载确认）
#   - 拖拽 + GrabCut 精修（魔棒替代）
#   - Shift 多选 + 合并选中框 + 建议合并
#   - ESC 关闭子窗口
import copy, os, shutil, threading, traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from core import (CLASS_NAMES, CLASS_NAME_TO_ID, CLASS_COLORS, IMG_EXTS,
                  STATUS_TEXT, STATUS_COLOR, parse_weak_label, judge_result,
                  magic_wand_rect, quick_select_rect, build_edge_map,
                  snap_rect_to_edges, dedupe_boxes,
                  merge_boxes_union, merge_boxes_wbf,
                  suggest_duplicate_groups,
                  grabcut_rect, fastsam_rect, fastsam_model_exists)


SHIFT_MASK = 0x0001


class GTTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.current_image_path = None
        self.current_image = None
        self.display_image = None
        self.photo = None
        self.detections = []

        # 多选
        self.selected_indices = []

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.history = []
        self.max_history = 50

        # 画框模式：rect=拖拽, sam=SAM点击分割
        self.draw_mode = tk.StringVar(value="rect")
        self.use_grabcut = tk.BooleanVar(value=False)
        self.snap_var = tk.BooleanVar(value=True)

        self.label_mode = tk.BooleanVar(value=False)
        self.manual_class_var = tk.StringVar(value="obstacle")
        self.drag_start = None

        self._edge_cache = None
        self._np_rgb_cache = None

        self.review_mode = False
        self.review_key = None

        self.gt_stat_var = tk.StringVar(value="GT: 0  |  已审核: 0  |  待审核: 0")
        self.cur_mode_var = tk.StringVar(value="当前: 未加载")

        self._build_ui()

    # ===== 刷新所有 GT 视图 =====
    def _refresh_all_gt_views(self):
        if hasattr(self.app, "refresh_all_gt_views"):
            try:
                self.app.refresh_all_gt_views()
                return
            except Exception as e:
                print(f"[刷新其他视图失败] {e}")
        self.refresh_gt_list()
        if hasattr(self.app, "review_tab"):
            try: self.app.review_tab.refresh_gt_list()
            except Exception: pass

    def _build_ui(self):
        app = self.app

        mb = ttk.LabelFrame(self, text="当前模型 / GT 库状态")
        mb.pack(fill=tk.X, padx=8, pady=(6, 2))
        self.model_info_var = tk.StringVar(value="未加载模型")
        ttk.Label(mb, textvariable=self.model_info_var,
                  font=('Consolas', 10), foreground="#007acc").pack(anchor=tk.W, padx=8, pady=4)
        ttk.Label(mb, textvariable=self.gt_stat_var,
                  font=('Consolas', 10), foreground="#009933").pack(anchor=tk.W, padx=8, pady=(0, 4))

        top = ttk.Frame(self); top.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(top, text="权重:").pack(side=tk.LEFT)
        self.pred_combo = ttk.Combobox(top, textvariable=app.weights_pred, values=[], width=35, state="readonly")
        self.pred_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="选择", command=self._choose_weights).pack(side=tk.LEFT)
        ttk.Button(top, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="  置信度:").pack(side=tk.LEFT)
        ttk.Scale(top, from_=0.05, to=0.95, variable=app.conf_thr,
                  orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=4)
        self.conf_label = ttk.Label(top, text=f"{app.conf_thr.get():.2f}")
        self.conf_label.pack(side=tk.LEFT)
        app.conf_thr.trace_add('write', lambda *a: self.conf_label.config(
            text=f"{app.conf_thr.get():.2f}"))

        top2 = ttk.Frame(self); top2.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(top2, text="上传图片", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top2, text="运行检测", command=self.run_detect).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(top2, text="补标模式", variable=self.label_mode,
                        command=self._toggle_mode).pack(side=tk.LEFT, padx=8)
        ttk.Label(top2, text="类别:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Combobox(top2, textvariable=self.manual_class_var,
                     values=["obstacle", "cola", "football"],
                     width=10, state="readonly").pack(side=tk.LEFT, padx=2)
        ttk.Label(top2, textvariable=self.cur_mode_var,
                  font=('Consolas', 10), foreground="#cc6600").pack(side=tk.RIGHT, padx=8)

        # ===== 画框模式 =====
        mode_row = ttk.Frame(self); mode_row.pack(fill=tk.X, padx=8, pady=(2, 0))
        ttk.Label(mode_row, text="画框模式:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(mode_row, text="拖拽", variable=self.draw_mode,
                        value="rect").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_row, text="SAM 点击分割", variable=self.draw_mode,
                        value="sam").pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(mode_row, text="拖拽后 GrabCut 精修",
                        variable=self.use_grabcut).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(mode_row, text="边缘吸附", variable=self.snap_var).pack(side=tk.LEFT, padx=8)

        # ===== 操作按钮 =====
        top3 = ttk.Frame(self); top3.pack(fill=tk.X, padx=8, pady=2)
        for txt, cb in [("删除选中框", self.delete_selected),
                        ("清空所有框", self.clear_boxes),
                        ("改选中框类别", self.apply_class),
                        ("↶ 撤销", self.undo),
                        ("🧹 去重", self.dedupe_current),
                        ("🔗 合并选中框", self.merge_selected),
                        ("🔍 建议合并", self.auto_suggest_merge),
                        ("⭐ 设为必要框", self.set_required),
                        ("☆ 取消必要框", self.unset_required),
                        ("💾 存入 GT 库", self.save_to_gt),
                        ("💾 存入训练集", self.save_to_training)]:
            ttk.Button(top3, text=txt, command=cb).pack(side=tk.LEFT, padx=3)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(main)
        self.canvas = tk.Canvas(left, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Configure>', lambda e: self.update_canvas())
        main.add(left, weight=4)

        right = ttk.Frame(main, width=420); main.add(right, weight=1)

        gtf = ttk.LabelFrame(right, text="GT 库列表（双击审核）")
        gtf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        filter_row = ttk.Frame(gtf); filter_row.pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Label(filter_row, text="筛选:").pack(side=tk.LEFT)
        self.gt_filter_var = tk.StringVar(value="全部")
        ttk.Combobox(filter_row, textvariable=self.gt_filter_var,
                     values=["全部", "待审核", "已审核", "未通过", "已通过"],
                     width=10, state="readonly").pack(side=tk.LEFT, padx=4)
        self.gt_filter_var.trace_add('write', lambda *a: self.refresh_gt_list())

        cols = ("key", "boxes", "req", "reviewed", "score", "status")
        self.gt_tree = ttk.Treeview(gtf, columns=cols, show="headings", height=18)
        for c, t, w in [("key", "图片", 150), ("boxes", "框", 40),
                        ("req", "必要", 40), ("reviewed", "审核", 45),
                        ("score", "F1", 45), ("status", "检测", 55)]:
            self.gt_tree.heading(c, text=t)
            self.gt_tree.column(c, width=w, anchor="center" if c != "key" else "w")
        sb = ttk.Scrollbar(gtf, orient="vertical", command=self.gt_tree.yview)
        self.gt_tree.configure(yscrollcommand=sb.set)
        self.gt_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        self.gt_tree.tag_configure("reviewed", foreground="#008800")
        self.gt_tree.tag_configure("unreviewed", foreground="#cc6600")
        self.gt_tree.tag_configure("pass", foreground="#008800")
        self.gt_tree.tag_configure("fail", foreground="#cc0000")
        self.gt_tree.tag_configure("merged", background="#e0f0ff")
        self.gt_tree.bind('<Double-Button-1>', self._on_gt_double)

        gtb = ttk.Frame(right); gtb.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(gtb, text="✓ 审核通过", command=self.review_approve).pack(side=tk.LEFT, padx=2)
        ttk.Button(gtb, text="✗ 未通过", command=self.review_reject).pack(side=tk.LEFT, padx=2)
        ttk.Button(gtb, text="📥 从检测更新", command=self.update_from_detections).pack(side=tk.LEFT, padx=2)

        self.refresh_model_info()
        self.refresh_gt_list()

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

    # ==================== 模型 ====================
    def _choose_weights(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if p: self.app.weights_pred.set(p)

    def refresh_model_info(self):
        try:
            if self.app.detector is None:
                self.model_info_var.set("未加载模型"); return
            path = Path(self.app.weights_pred.get())
            if not path.exists():
                self.model_info_var.set("路径不存在"); return
            from datetime import datetime
            mt = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            sz = path.stat().st_size / 1024 / 1024
            names = list(self.app.detector.names.values())
            self.model_info_var.set(
                f"✓ {path.name}  |  {sz:.1f} MB  |  {mt}  |  {', '.join(map(str, names))}")
        except Exception as e:
            self.model_info_var.set(f"错误: {e}")

    def refresh_gt_stat(self):
        try:
            gt = self.app.ground_truth
            merge_n = self.app.merge_log.count() if hasattr(self.app, "merge_log") else 0
            guidance_n = self.app.guidance.count() if hasattr(self.app, "guidance") else 0
            self.gt_stat_var.set(
                f"GT 库: {gt.count()} 张  |  "
                f"已审核: {gt.count_reviewed()}  |  "
                f"待审核: {gt.count_unreviewed()}  |  "
                f"模型通过: {gt.count_pass()}  |  "
                f"模型未通过: {gt.count_fail()}  |  "
                f"人工合并: {merge_n}  |  人工引导: {guidance_n}")
        except Exception:
            pass

    def load_model(self):
        path = self.app.weights_pred.get()
        if not Path(path).exists():
            messagebox.showerror("错误", f"权重不存在:\n{path}"); return
        try:
            from predict import Detector
            self.app.detector = Detector(path)
            names = list(self.app.detector.names.values())
            self.app.status.set(f"已加载: {Path(path).name}")
            self.refresh_model_info()
            messagebox.showinfo("成功", f"类别: {', '.join(map(str, names))}")
        except Exception as e:
            self.app.detector = None
            self.refresh_model_info()
            messagebox.showerror("失败", f"{type(e).__name__}: {e}")

    def try_auto_load(self):
        p = self.app.weights_pred.get()
        if not p or not Path(p).exists(): return
        try:
            from predict import Detector
            self.app.detector = Detector(p)
            self.app.status.set(f"已自动加载: {Path(p).name}")
        except Exception as e:
            self.app.detector = None
            print(f"[自动加载失败] {e}")

    # ==================== 缓存 ====================
    def _build_edge_cache(self):
        try:
            if self.current_image is None:
                self._edge_cache = None
                self._np_rgb_cache = None
                return
            img = np.array(self.current_image)
            self._np_rgb_cache = img
            bgr = img[:, :, ::-1].copy()
            self._edge_cache = build_edge_map(bgr)
        except Exception as e:
            print(f"[边缘缓存失败] {e}")
            self._edge_cache = None
            self._np_rgb_cache = None

    # ==================== 图片加载 ====================
    def load_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not p: return
        try:
            self.review_mode = False; self.review_key = None
            self.cur_mode_var.set(f"当前: {Path(p).name} (新上传)")
            self.current_image_path = p
            self.current_image = Image.open(p).convert("RGB")
            self.detections = []
            self.selected_indices = []
            self.display_image = self.current_image.copy()
            self._build_edge_cache()
            self.update_canvas()
            self.history.clear()
            if self.app.detector is None: self.try_auto_load()
            self.refresh_model_info()
            if self.app.detector is not None:
                self.run_detect()
            self.app.status.set(f"已加载: {Path(p).name}")
        except Exception as e:
            messagebox.showerror("加载失败", f"{e}")

    def run_detect(self):
        if self.app.detector is None:
            messagebox.showwarning("提示", "请先加载模型"); return
        if self.current_image is None:
            messagebox.showwarning("提示", "请先上传图片"); return
        conf = float(self.app.conf_thr.get())
        self.app.status.set("检测中..."); self.update_idletasks()
        try:
            raw = self.app.detector.predict(self.current_image, conf=conf)
        except Exception as e:
            print(traceback.format_exc())
            messagebox.showerror("检测失败", f"{type(e).__name__}: {e}"); return
        self.detections = [dict(d) for d in raw]
        for d in self.detections:
            d["manual"] = False
            d["required"] = False
        self.selected_indices = []
        self.render()
        self.app.status.set(f"检测完成，{len(self.detections)} 个目标（Shift 多选可合并）")

    # ==================== GT 列表 ====================
    def refresh_gt_list(self):
        try:
            for iid in self.gt_tree.get_children(): self.gt_tree.delete(iid)
            gt = self.app.ground_truth
            merge_log = getattr(self.app, "merge_log", None)
            filt = self.gt_filter_var.get()
            for key, rec in gt.all_items():
                reviewed = rec.get("reviewed", False)
                status = rec.get("status", "untested")
                score = rec.get("last_score")
                if filt == "待审核" and reviewed: continue
                if filt == "已审核" and not reviewed: continue
                if filt == "未通过" and status != "fail": continue
                if filt == "已通过" and status != "pass": continue
                req_n = sum(1 for b in rec.get("boxes", []) if b.get("required", False))
                tags = []
                if reviewed: tags.append("reviewed"); rev_txt = "✓"
                else: tags.append("unreviewed"); rev_txt = "⚠"
                # ===== 状态只看 F1 =====
                if status == "img_missing":
                    st_txt = "图片缺失"
                elif score is None:
                    st_txt = "-"
                elif score >= 0.9 - 1e-6:
                    st_txt = "✓通过"
                    tags.append("pass")
                else:
                    st_txt = "✗未通过"
                    tags.append("fail")
                if merge_log and merge_log.is_merged(key):
                    tags.append("merged")
                sc_txt = f"{score:.2f}" if score is not None else "---"
                self.gt_tree.insert("", "end", iid=key,
                    values=(key, len(rec.get("boxes", [])), req_n, rev_txt, sc_txt, st_txt),
                    tags=tuple(tags))
            self.refresh_gt_stat()
        except Exception as e:
            print(f"[GT列表刷新失败] {e}")

    def _on_gt_double(self, event):
        sel = self.gt_tree.selection()
        if not sel: return
        key = sel[0]
        rec = self.app.ground_truth.get_by_key(key)
        if not rec: return
        img_path = self.app.ground_truth.img_dir / rec["file"]
        if not img_path.exists():
            messagebox.showerror("错误", f"GT 图片丢失: {rec['file']}"); return
        self._load_gt_for_review(key, img_path, rec["boxes"])

    def _load_gt_for_review(self, key, img_path, boxes):
        try:
            self.review_mode = True; self.review_key = key
            self.current_image_path = str(img_path)
            self.current_image = Image.open(img_path).convert("RGB")
            self.detections = []
            for b in boxes:
                self.detections.append({
                    "xyxy": [float(v) for v in b["xyxy"]],
                    "cls_id": int(b["cls_id"]),
                    "cls_name": b["cls_name"],
                    "conf": 1.0,
                    "manual": False,
                    "required": bool(b.get("required", False)),
                })
            self.selected_indices = []
            self.display_image = self.current_image.copy()
            self._build_edge_cache()
            self.update_canvas()
            self.render()
            self.history.clear()
            self.cur_mode_var.set(f"当前: {key}  (GT 审核模式)")
            self.app.status.set(f"GT 审核: {key}（Shift 多选可合并）")
        except Exception as e:
            messagebox.showerror("加载失败", f"{e}")

    # ==================== 审核 ====================
    def review_approve(self):
        if not self.review_mode or not self.review_key:
            messagebox.showinfo("提示", "请先双击列表中的 GT 进入审核模式"); return
        try:
            self.app.ground_truth.update_boxes(self.review_key, self.detections, reviewed=True)
            self._refresh_all_gt_views()
            self.app.status.set(f"✓ 已审核: {self.review_key}")
        except Exception as e:
            messagebox.showerror("失败", f"{e}")

    def review_reject(self):
        if not self.review_mode or not self.review_key:
            messagebox.showinfo("提示", "请先双击列表中的 GT 进入审核模式"); return
        try:
            self.app.ground_truth.update_boxes(self.review_key, self.detections, reviewed=False)
            self.app.ground_truth.set_reviewed(self.review_key, False)
            self._refresh_all_gt_views()
            self.app.status.set(f"✗ 已标记待修: {self.review_key}")
        except Exception as e:
            messagebox.showerror("失败", f"{e}")

    def update_from_detections(self):
        if not self.review_mode or not self.review_key:
            messagebox.showinfo("提示", "请先进入 GT 审核模式"); return
        if self.app.detector is None:
            messagebox.showwarning("提示", "请先加载模型"); return
        conf = float(self.app.conf_thr.get())
        try:
            raw = self.app.detector.predict(self.current_image, conf=conf)
        except Exception as e:
            messagebox.showerror("检测失败", f"{e}"); return
        self.detections = [dict(d) for d in raw]
        for d in self.detections:
            d["manual"] = False
            d["required"] = False
        self.selected_indices = []
        self.render()
        self.app.ground_truth.update_boxes(self.review_key, self.detections, reviewed=False)
        self._refresh_all_gt_views()
        self.app.status.set(f"已用检测结果更新 GT: {self.review_key}")

    # ==================== 绘制 ====================
    def render(self):
        if self.current_image is None:
            return
        img = self.current_image.copy()
        draw = ImageDraw.Draw(img)
        iw, ih = img.size
        fs = max(12, min(20, iw // 60))
        font = self._get_font(fs)

        sel_set = set(self.selected_indices)

        for i, d in enumerate(self.detections):
            x1, y1, x2, y2 = d["xyxy"]
            is_req = d.get("required", False)
            if i in sel_set:
                color, width = '#FFFF00', 4
            elif is_req:
                color, width = '#FF00FF', 4
            elif d.get("manual"):
                color, width = '#00FFFF', 3
            else:
                color, width = CLASS_COLORS[d["cls_id"] % len(CLASS_COLORS)], 2

            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            tag = ""
            if is_req: tag += " ★"
            elif d.get("manual"): tag += " *"
            if i in sel_set: tag += " [选]"
            label = d['cls_name'] + tag

            try:
                bbox = draw.textbbox((0, 0), label, font=font)
            except Exception:
                bbox = (0, 0, len(label) * fs, fs)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            ty = y1 - th - 4
            if ty < 0: ty = y1 + 2
            tx = x1
            if tx + tw + 4 > iw: tx = iw - tw - 4
            if tx < 0: tx = 0
            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=color)
            draw.text((tx + 2, ty + 2), label, fill='black', font=font)

        # 顶部提示条
        n_sel = len(self.selected_indices)
        tip = f"已选 {n_sel} 个框  |  Shift+点击 多选，点「🔗 合并选中框」合并" if n_sel >= 2 \
              else "Shift+点击 可多选（用于合并）"
        try:
            bbox = draw.textbbox((0, 0), tip, font=font)
        except Exception:
            bbox = (0, 0, 400, 24)
        tw = bbox[2] - bbox[0] + 16
        draw.rectangle([5, 5, 5 + tw, 34], fill='#0088cc')
        draw.text((13, 11), tip, fill='white', font=font)

        if self.review_mode:
            banner = "GT 审核模式"
            try:
                bbox = draw.textbbox((0, 0), banner, font=font)
            except Exception:
                bbox = (0, 0, 200, 30)
            tw = bbox[2] - bbox[0] + 20
            draw.rectangle([5, 40, 5 + tw, 72], fill='#cc6600')
            draw.text((15, 46), banner, fill='white', font=font)

        self.display_image = img
        self.update_canvas()

    def update_canvas(self):
        if self.display_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            self.after(100, self.update_canvas)
            return
        iw, ih = self.display_image.size
        self.scale = min(cw / iw, ch / ih)
        nw = max(1, int(iw * self.scale))
        nh = max(1, int(ih * self.scale))
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2
        resized = self.display_image.resize((nw, nh), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)

    # ==================== 交互 ====================
    def _toggle_mode(self):
        self.canvas.config(cursor="cross" if self.label_mode.get() else "")

    def _hit_test(self, ix, iy):
        hit = None
        for i in range(len(self.detections) - 1, -1, -1):
            x1, y1, x2, y2 = self.detections[i]["xyxy"]
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                hit = i
                break
        return hit

    def _on_press(self, event):
        if self.label_mode.get() and self.current_image is not None:
            ix = (event.x - self.offset_x) / self.scale
            iy = (event.y - self.offset_y) / self.scale
            iw, ih = self.current_image.size
            if not (0 <= ix < iw and 0 <= iy < ih):
                return
            if self.draw_mode.get() == "sam":
                self._do_sam_click(ix, iy)
                return
            self.drag_start = (ix, iy)
            return

        if not self.detections:
            return
        ix = (event.x - self.offset_x) / self.scale
        iy = (event.y - self.offset_y) / self.scale
        hit = self._hit_test(ix, iy)

        is_shift = bool(event.state & SHIFT_MASK)
        if hit is None:
            if not is_shift:
                self.selected_indices = []
        else:
            if is_shift:
                if hit in self.selected_indices:
                    self.selected_indices.remove(hit)
                else:
                    self.selected_indices.append(hit)
            else:
                self.selected_indices = [hit]
        self.render()

    # ==================== SAM 点击分割 ====================
    def _do_sam_click(self, ix, iy):
        try:
            img = self._np_rgb_cache
            if img is None:
                img = np.array(self.current_image)
                self._np_rgb_cache = img

            # 1. 检查 FastSAM 模型是否已存在
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
                # 回退 GrabCut
                if messagebox.askyesno("SAM 未圈到",
                        "SAM 没有圈出物体。\n是否改用 GrabCut（需要先拖一个大概的框）？"):
                    return
                return

            x1, y1, x2, y2 = rect
            if self.snap_var.get() and self._edge_cache is not None:
                x1, y1, x2, y2 = snap_rect_to_edges(
                    self._edge_cache, x1, y1, x2, y2, search=6, min_hit=2)

            self._push_hist()
            cls_name = self.manual_class_var.get()
            cid = CLASS_NAME_TO_ID[cls_name]
            self.detections.append({
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": 1.0,
                "cls_id": cid,
                "cls_name": cls_name,
                "manual": True,
                "required": False,
            })
            self.render()
            self.app.status.set(
                f"SAM 添加: {cls_name}  {int(x2 - x1)}×{int(y2 - y1)}")
        except Exception as e:
            print(traceback.format_exc())
            self.app.status.set(f"SAM 失败: {e}")
            messagebox.showerror("SAM 失败",
                f"{type(e).__name__}: {e}\n\n可以改用拖拽 + GrabCut。")

    # ==================== 拖拽 ====================
    def _on_drag(self, event):
        if not self.label_mode.get() or self.drag_start is None:
            return
        if self.draw_mode.get() == "sam":
            return
        ix = (event.x - self.offset_x) / self.scale
        iy = (event.y - self.offset_y) / self.scale
        x1, y1 = self.drag_start
        cx1 = self.offset_x + x1 * self.scale
        cy1 = self.offset_y + y1 * self.scale
        cx2 = self.offset_x + ix * self.scale
        cy2 = self.offset_y + iy * self.scale
        self.canvas.delete("drag_rect")
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                      outline='#00ff00', width=3, tags="drag_rect")

    def _on_release(self, event):
        if not self.label_mode.get(): return
        if self.draw_mode.get() == "sam": return
        if self.drag_start is None: return
        ix = (event.x - self.offset_x) / self.scale
        iy = (event.y - self.offset_y) / self.scale
        x1, y1 = self.drag_start
        x2, y2 = ix, iy
        self.drag_start = None
        self.canvas.delete("drag_rect")
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5: return
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        iw, ih = self.current_image.size
        x1 = max(0, min(x1, iw)); x2 = max(0, min(x2, iw))
        y1 = max(0, min(y1, ih)); y2 = max(0, min(y2, ih))

        # GrabCut 精修
        if self.use_grabcut.get() and self._np_rgb_cache is not None:
            try:
                self.app.status.set("GrabCut 精修中...")
                self.update_idletasks()
                x1, y1, x2, y2 = grabcut_rect(self._np_rgb_cache, (x1, y1, x2, y2))
            except Exception as e:
                print(f"[GrabCut 失败] {e}")

        if self.snap_var.get() and self._edge_cache is not None:
            try:
                x1, y1, x2, y2 = snap_rect_to_edges(
                    self._edge_cache, x1, y1, x2, y2, search=10, min_hit=3)
            except Exception as e:
                print(f"[吸附失败] {e}")

        self._push_hist()
        cls_name = self.manual_class_var.get()
        cid = CLASS_NAME_TO_ID[cls_name]
        self.detections.append({
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "conf": 1.0,
            "cls_id": cid,
            "cls_name": cls_name,
            "manual": True,
            "required": False,
        })
        self.render()

    def _push_hist(self):
        self.history.append(copy.deepcopy(self.detections))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self):
        if not self.history: return
        self.detections = self.history.pop()
        self.selected_indices = []
        self.render()

    def apply_class(self):
        if not self.selected_indices:
            messagebox.showinfo("提示", "请先选框"); return
        self._push_hist()
        name = self.manual_class_var.get()
        cid = CLASS_NAME_TO_ID[name]
        for i in self.selected_indices:
            if 0 <= i < len(self.detections):
                self.detections[i]["cls_id"] = cid
                self.detections[i]["cls_name"] = name
        self.render()

    def set_required(self):
        if not self.selected_indices:
            messagebox.showinfo("提示", "请先选择一个框"); return
        self._push_hist()
        for i in self.selected_indices:
            if 0 <= i < len(self.detections):
                self.detections[i]["required"] = True
        self.render()
        self.app.status.set(f"已设为必要框: {len(self.selected_indices)} 个")

    def unset_required(self):
        if not self.selected_indices:
            messagebox.showinfo("提示", "请先选择一个框"); return
        self._push_hist()
        for i in self.selected_indices:
            if 0 <= i < len(self.detections):
                self.detections[i]["required"] = False
        self.render()
        self.app.status.set(f"已取消必要框: {len(self.selected_indices)} 个")

    def delete_selected(self):
        if not self.selected_indices:
            messagebox.showinfo("提示", "请先选框"); return
        n = len(self.selected_indices)
        if not messagebox.askyesno("确认", f"删除选中的 {n} 个框？"):
            return
        self._push_hist()
        for i in sorted(self.selected_indices, reverse=True):
            if 0 <= i < len(self.detections):
                del self.detections[i]
        self.selected_indices = []
        self.render()

    def clear_boxes(self):
        if not self.detections: return
        if messagebox.askyesno("确认", f"清空 {len(self.detections)} 个框？"):
            self._push_hist()
            self.detections = []
            self.selected_indices = []
            self.render()

    def dedupe_current(self):
        if not self.detections:
            messagebox.showinfo("提示", "当前没有框"); return
        keep, rejected = dedupe_boxes(self.detections)
        if not rejected:
            messagebox.showinfo("提示", f"没有发现重复框（{len(self.detections)} 个）")
            return
        self._push_hist()
        self.detections = keep
        self.selected_indices = []
        self.render()
        msg = f"删除 {len(rejected)} 个重复框，保留 {len(keep)} 个"
        self.app.status.set(msg)
        messagebox.showinfo("完成", msg)

    # ==================== 合并 ====================
    def _current_img_name(self):
        if not self.current_image_path:
            return None
        if self.review_mode and self.review_key:
            return self.review_key
        return Path(self.current_image_path).name

    def _record_merge(self, before_n, after_n, method):
        name = self._current_img_name()
        if not name:
            return
        try:
            self.app.merge_log.record(name, before_n, after_n, method)
        except Exception as e:
            print(f"[合并记录失败] {e}")

    def merge_selected(self):
        if len(self.selected_indices) < 2:
            messagebox.showinfo("提示", "请按住 Shift 多选至少两个框")
            return
        indices = sorted(self.selected_indices)
        boxes = [self.detections[i] for i in indices
                 if 0 <= i < len(self.detections)]
        if len(boxes) < 2:
            return

        cls_ids = set(b["cls_id"] for b in boxes)
        if len(cls_ids) > 1:
            if not messagebox.askyesno(
                "类别不同",
                "选中的框类别不同，确定合并吗？\n"
                "合并后类别取置信度最高的那个。"):
                return
            method = "wbf"
            merged = merge_boxes_wbf(boxes)
        else:
            answer = messagebox.askyesnocancel(
                "合并方式",
                "是否使用「加权融合(WBF)」？\n\n"
                "「是」= 加权融合（推荐，框更准）\n"
                "「否」= 并集外接矩形（保守，框更大）\n"
                "「取消」= 放弃合并")
            if answer is None:
                return
            method = "wbf" if answer else "union"
            merged = merge_boxes_wbf(boxes) if answer else merge_boxes_union(boxes)
        if merged is None:
            return

        self._push_hist()
        for i in sorted(indices, reverse=True):
            if 0 <= i < len(self.detections):
                del self.detections[i]
        self.detections.append(merged)
        self.selected_indices = []
        self._record_merge(len(boxes), 1, method)
        self.render()
        self.app.status.set(f"已合并 {len(boxes)} 个框 ({method})")
        self.refresh_gt_stat()

    def auto_suggest_merge(self):
        if len(self.detections) < 2:
            messagebox.showinfo("提示", "当前框不足 2 个")
            return
        groups = suggest_duplicate_groups(self.detections, iou_thr=0.55,
                                          contain_thr=0.80)
        if not groups:
            messagebox.showinfo("提示", "没发现建议合并的重复框")
            return
        all_idx = []
        for g in groups:
            all_idx.extend(g)
        self.selected_indices = sorted(set(all_idx))
        self.render()
        messagebox.showinfo(
            "建议合并",
            f"发现 {len(groups)} 组重复框，已全部选中。\n"
            f"共 {len(self.selected_indices)} 个框。\n\n"
            f"点「🔗 合并选中框」逐组合并，\n"
            f"或按住 Shift 手动调整选中。")

    # ==================== 保存 ====================
    def save_to_gt(self):
        if self.current_image is None or not self.current_image_path:
            messagebox.showwarning("提示", "请先上传图片"); return
        if not self.detections:
            messagebox.showwarning("提示", "当前没有框"); return
        try:
            if self.review_mode and self.review_key:
                self.app.ground_truth.update_boxes(self.review_key, self.detections, reviewed=True)
                self._refresh_all_gt_views()
                messagebox.showinfo("已保存",
                    f"已更新 GT 并标记为已审核\n框数: {len(self.detections)}")
                return
            self.app.ground_truth.add(self.current_image_path, self.detections, reviewed=False)
            self._refresh_all_gt_views()
            messagebox.showinfo("已保存",
                f"已加入 GT 库（待审核）\n框数: {len(self.detections)}\n\n"
                f"在右侧列表双击该图可进入审核模式")
        except Exception as e:
            messagebox.showerror("失败", f"{e}")

    def save_to_training(self):
        try:
            if self.current_image is None or not self.current_image_path:
                raise RuntimeError("没图片")
            if not self.detections:
                raise RuntimeError("没有框")
            img_dir = Path(self.app.auto_img_dir.get())
            lbl_dir = Path(self.app.auto_labels_dir.get())
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            dst_img = img_dir / Path(self.current_image_path).name
            if not dst_img.exists():
                shutil.copy2(self.current_image_path, dst_img)
            iw, ih = self.current_image.size
            with open(lbl_dir / (Path(self.current_image_path).stem + ".txt"),
                      "w", encoding="utf-8") as f:
                for d in self.detections:
                    x1, y1, x2, y2 = d["xyxy"]
                    cx = (x1 + x2) / 2 / iw
                    cy = (y1 + y2) / 2 / ih
                    w = (x2 - x1) / iw
                    h = (y2 - y1) / ih
                    f.write(f"{d['cls_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            self.app.status.set(f"已保存到训练集: {Path(self.current_image_path).name}")
            messagebox.showinfo("保存成功", f"框数: {len(self.detections)}")
        except Exception as e:
            messagebox.showwarning("提示", str(e))