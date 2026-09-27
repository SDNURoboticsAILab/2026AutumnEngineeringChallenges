# tab_aug.py
import os, random, threading, traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import numpy as np

from core import _read_labels, _write_labels, get_augmentations


class AugTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.factor = tk.IntVar(value=5)
        self.clear_old = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="未执行")
        self._build_ui()

    def _build_ui(self):
        app = self.app
        info = ttk.LabelFrame(self, text="数据增强说明"); info.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(info, text="对 dataset/images/train 每张原图生成多张变换图（颜色/翻转/噪声/裁剪）",
                  justify=tk.LEFT, font=('Consolas', 10)).pack(anchor=tk.W, padx=8, pady=6)

        params = ttk.LabelFrame(self, text="增强参数"); params.pack(fill=tk.X, padx=10, pady=4)
        row = ttk.Frame(params); row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row, text="图片目录:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=app.auto_img_dir, width=60).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="选择", command=self._choose_img).pack(side=tk.LEFT)
        row2 = ttk.Frame(params); row2.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row2, text="标签目录:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=app.auto_labels_dir, width=60).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="选择", command=self._choose_lbl).pack(side=tk.LEFT)
        row3 = ttk.Frame(params); row3.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row3, text="增强倍数:").pack(side=tk.LEFT)
        ttk.Spinbox(row3, from_=1, to=9, textvariable=self.factor, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(row3, text="张/原图").pack(side=tk.LEFT)
        ttk.Checkbutton(row3, text="先清空旧的 aug_", variable=self.clear_old).pack(side=tk.LEFT, padx=20)

        btns = ttk.Frame(self); btns.pack(fill=tk.X, padx=10, pady=6)
        self.btn_start = ttk.Button(btns, text="开始增强", command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="打开图片目录",
                   command=lambda: os.startfile(app.auto_img_dir.get())
                   if Path(app.auto_img_dir.get()).exists() else None).pack(side=tk.LEFT, padx=5)
        ttk.Label(btns, textvariable=self.status_var, foreground="#007acc").pack(side=tk.RIGHT, padx=6)

        logf = ttk.LabelFrame(self, text="增强日志"); logf.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        sb = ttk.Scrollbar(logf); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log = tk.Text(logf, wrap=tk.WORD, yscrollcommand=sb.set,
                           font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4')
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb.config(command=self.log.yview)
        self.log.config(state=tk.DISABLED)

    def _choose_img(self):
        p = filedialog.askdirectory()
        if p: self.app.auto_img_dir.set(p)
    def _choose_lbl(self):
        p = filedialog.askdirectory()
        if p: self.app.auto_labels_dir.set(p)

    def _log_msg(self, msg):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n"); self.log.see(tk.END)
        self.log.config(state=tk.DISABLED); self.update_idletasks()

    def start(self):
        img_dir = Path(self.app.auto_img_dir.get()); lbl_dir = Path(self.app.auto_labels_dir.get())
        if not img_dir.exists() or not lbl_dir.exists():
            messagebox.showerror("错误", "目录不存在"); return
        self.btn_start.config(state=tk.DISABLED)
        self.status_var.set("增强中...")
        self.log.config(state=tk.NORMAL); self.log.delete('1.0', tk.END); self.log.config(state=tk.DISABLED)
        threading.Thread(target=self._do_aug,
                         args=(img_dir, lbl_dir, self.factor.get(), self.clear_old.get()),
                         daemon=True).start()

    def _do_aug(self, img_dir, lbl_dir, factor, clear_old):
        try:
            random.seed(42); np.random.seed(42)
            if clear_old:
                for p in img_dir.glob("aug_*"): p.unlink()
                for p in lbl_dir.glob("aug_*.txt"): p.unlink()
                self._log_msg("已清空旧的 aug_ 文件")
            originals = [p for p in img_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")
                         and not p.name.startswith("aug_")]
            self._log_msg(f"发现 {len(originals)} 张原图")
            augmentations = get_augmentations()
            total = 0
            for i, img_path in enumerate(originals, 1):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                boxes = _read_labels(lbl_path)
                img = cv2.imread(str(img_path))
                if img is None: continue
                chosen = random.sample(augmentations, min(factor, len(augmentations)))
                for name, fn in chosen:
                    try:
                        new_img, new_boxes = fn(img, boxes)
                        new_name = f"aug_{name}_{img_path.stem}"
                        cv2.imwrite(str(img_dir / f"{new_name}{img_path.suffix}"), new_img)
                        _write_labels(lbl_dir / f"{new_name}.txt", new_boxes)
                        total += 1
                    except Exception as e: self._log_msg(f"  失败 {name}: {e}")
                if i % 20 == 0 or i == len(originals):
                    self._log_msg(f"[进度] {i}/{len(originals)}  已生成 {total}")
                    self.after(0, lambda c=total: self.status_var.set(f"增强中... {c} 张"))
            self._log_msg(f"\n✅ 完成！共 {total} 张")
            self.status_var.set(f"完成: +{total}")
            self.after(0, lambda: messagebox.showinfo("完成", f"共 {total} 张"))
        except Exception as e:
            self._log_msg(f"\n❌ {traceback.format_exc()}")
            self.after(0, lambda: messagebox.showerror("失败", str(e)))
        finally:
            self.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
