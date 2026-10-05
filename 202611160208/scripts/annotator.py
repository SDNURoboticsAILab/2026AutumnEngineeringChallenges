# -*- coding: utf-8 -*-
"""YOLO 标注修正器 v2.2（tkinter，零 Qt 依赖，不闪退）

用法:
  python scripts/annotator.py --split val        # 数据集 val（默认，186 张）
  python scripts/annotator.py --split train      # 数据集 train（749 张）
  python scripts/annotator.py --dir D:\\pics      # 任意图片文件夹（标签默认存其下 labels）
  或双击项目根目录 启动标注.bat

操作:
  1/2/3    选类别        拖拽左键   画新框
  单击框    选中(红)      Del       删除选中     双击框    直接删除
  A / D    上一张/下一张   F         下一张未标注
  Tab      循环选框       Ctrl+S    保存（右侧可勾选自动保存）
"""
import argparse
import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")
PROGRESS_FILE = os.path.join(ROOT, "yolo_project", "annotation_progress.json")
CLASSES = ["obstacle", "cola", "football"]
COLORS = {0: "#ff8c00", 1: "#00c800", 2: "#2060ff"}


class Annotator:
    def __init__(self, split, img_dir=None, lbl_dir=None):
        self.split = split
        self.default_img_dir = os.path.join(DATASET, "images", split)
        self.img_dir = img_dir or self.default_img_dir
        self.lbl_dir = lbl_dir or self._default_labels_for(self.img_dir)
        self.idx = 0
        self.cur_class = 0
        self.boxes = []
        self.selected = None
        self.dirty = False
        self.drag_start = None
        self.rect_id = None
        self.img = None
        self.photo = None
        self.canvas_w, self.canvas_h = 1000, 700
        self._file_lock = False          # 程序刷新文件列表时屏蔽点击事件

        # 进度恢复
        self.progress = self._load_progress()
        start = self._use_folder(self.img_dir, self.lbl_dir, strict=True)

        self.root = tk.Tk()
        try:
            self.root.state("zoomed")          # Windows 启动即最大化
        except tk.TclError:
            self.root.geometry("1200x760")
        self._build_ui()
        self._update_toolbar()
        self.root.bind("<KeyPress>", self.on_key)
        self.load_image(start)

    # ---------- 文件夹与标签目录 ----------
    @staticmethod
    def _default_labels_for(img_dir):
        """推算标签保存目录：dataset/images/val -> dataset/labels/val；其他 -> <图片目录>/labels"""
        parent = os.path.dirname(img_dir)
        name = os.path.basename(img_dir.rstrip("\\/"))
        if os.path.basename(parent).lower() == "images":
            return os.path.join(parent, "labels", name)
        return os.path.join(img_dir, "labels")

    def _folder_key(self):
        """进度文件的键：数据集目录沿用 train/val，其他文件夹用完整路径"""
        if os.path.normcase(self.img_dir) == os.path.normcase(self.default_img_dir):
            return self.split
        return os.path.normcase(os.path.normpath(self.img_dir))

    def _use_folder(self, img_dir, lbl_dir=None, strict=False):
        """切换图片文件夹并恢复该文件夹的进度，返回起始下标；失败时返回 None（strict 则报错退出）"""
        d = os.path.normpath(img_dir)
        if not os.path.isdir(d):
            if strict:
                raise SystemExit(f"图片目录不存在: {d}")
            return None
        files = sorted(f for f in os.listdir(d)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not files:
            if strict:
                raise SystemExit(f"目录无图片: {d}")
            return None
        self.img_dir = d
        self.lbl_dir = os.path.normpath(lbl_dir) if lbl_dir else self._default_labels_for(d)
        self.files = files
        info = self.progress.get(self._folder_key(), {})
        self.done = set(info.get("done", []))
        start = info.get("last_index", 0)
        return min(max(start, 0), len(self.files) - 1)

    # ---------- 进度持久化 ----------
    def _load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception:
                pass
        return {"train": {}, "val": {}}

    def _persist(self):
        self.progress[self._folder_key()] = {
            "last_index": self.idx,
            "done": sorted(self.done),
        }
        self.progress["config"] = {
            "last_dir": self.img_dir,
            "last_save": self.lbl_dir,
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as fp:
            json.dump(self.progress, fp, ensure_ascii=False, indent=1)

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side="top", fill="x", padx=8, pady=(6, 2))
        ttk.Button(top, text="打开图片文件夹…", command=self.open_folder).pack(side="left")
        ttk.Button(top, text="更改保存文件夹…", command=self.change_save_dir).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="下一张未标注 (F)", command=self.next_unlabeled).pack(side="left", padx=(6, 0))
        self.save_lbl = tk.Label(top, text="", fg="#2060ff")
        self.save_lbl.pack(side="left", padx=12)

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, bg="black", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)

        side = ttk.Frame(main, width=240)
        side.pack(side="right", fill="y", padx=8, pady=4)

        self.cur_lbl = ttk.Label(side, text="", font=("Microsoft YaHei", 12, "bold"))
        self.cur_lbl.pack(pady=(0, 6))

        self.class_btns = []
        for i, c in enumerate(CLASSES):
            b = tk.Button(side, text=f"{i} {c}", width=20, height=1,
                          bg=COLORS[i], fg="white", font=("Microsoft YaHei", 10))
            b.config(command=lambda i=i: self.set_class(i))
            b.pack(pady=3)
            self.class_btns.append(b)

        ttk.Separator(side).pack(fill="x", pady=8)
        self.autosave_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="自动保存(增删框即存盘)",
                        variable=self.autosave_var).pack(anchor="w")
        ttk.Button(side, text="◀ 上一张 (A)", command=self.prev).pack(pady=2, fill="x")
        ttk.Button(side, text="下一张 ▶ (D)", command=self.next).pack(pady=2, fill="x")
        ttk.Button(side, text="删除选中框 (Del)", command=self.delete_selected).pack(pady=6, fill="x")
        ttk.Button(side, text="保存当前图 (Ctrl+S)", command=self.save).pack(pady=2, fill="x")

        ttk.Label(side, text="框列表(点选=选中):").pack(anchor="w")
        self.listbox = tk.Listbox(side, width=26, height=7)
        self.listbox.pack(pady=4, fill="x")
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        ttk.Label(side, text="文件列表(✔=已保存,点选跳图):").pack(anchor="w")
        self.file_list = tk.Listbox(side, width=26)
        self.file_list.pack(pady=(2, 4), fill="both", expand=True)
        self.file_list.bind("<<ListboxSelect>>", self.on_file_select)

        self.status = tk.Label(self.root, text="", anchor="w", relief="sunken")
        self.status.pack(fill="x", side="bottom")

    def set_class(self, i):
        self.cur_class = i
        for k, b in enumerate(self.class_btns):
            b.config(font=("Microsoft YaHei", 10, "bold" if k == i else "normal"),
                     relief="raised" if k == i else "flat",
                     bd=3 if k == i else 1)
        self.cur_lbl.config(text=f"当前类别: {i} {CLASSES[i]}", foreground=COLORS[i])
        self.redraw()

    # ---------- 尺寸自适应 ----------
    def on_resize(self, e):
        if abs(e.width - self.canvas_w) < 6 and abs(e.height - self.canvas_h) < 6:
            return
        self.canvas_w, self.canvas_h = e.width, e.height
        self.render_photo()
        self.redraw()

    def render_photo(self):
        if self.img is None:
            return
        scale = min(self.canvas_w / self.img_w, self.canvas_h / self.img_h)
        self.disp_w = max(1, int(self.img_w * scale))
        self.disp_h = max(1, int(self.img_h * scale))
        self.ox = (self.canvas_w - self.disp_w) // 2
        self.oy = (self.canvas_h - self.disp_h) // 2
        self.photo = ImageTk.PhotoImage(self.img.resize((self.disp_w, self.disp_h)))

    # ---------- 图片加载/绘制 ----------
    def load_image(self, idx):
        self.idx = idx % len(self.files)
        fname = self.files[self.idx]
        self.img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        self.img_w, self.img_h = self.img.size
        self.render_photo()
        self.boxes = self.read_label(os.path.splitext(fname)[0])
        self.selected = None
        self.dirty = False
        self.redraw()
        self._refresh_files()
        self._persist()

    def read_label(self, stem):
        lp = os.path.join(self.lbl_dir, stem + ".txt")
        boxes = []
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) == 5:
                    boxes.append((int(p[0]), *map(float, p[1:])))
        return boxes

    def redraw(self):
        self.canvas.delete("all")
        if self.photo:
            self.canvas.create_image(self.ox, self.oy, anchor="nw", image=self.photo)
        for i, (cid, cx, cy, w, h) in enumerate(self.boxes):
            x1 = self.ox + (cx - w / 2) * self.disp_w
            y1 = self.oy + (cy - h / 2) * self.disp_h
            x2 = self.ox + (cx + w / 2) * self.disp_w
            y2 = self.oy + (cy + h / 2) * self.disp_h
            col = COLORS.get(cid, "red")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=col,
                                         width=4 if i == self.selected else 2)
            self.canvas.create_text(x1 + 2, y1 - 7, anchor="w", text=CLASSES[cid],
                                    fill=col, font=("Arial", 10, "bold"))
        self.refresh_list()
        fname = self.files[self.idx]
        mark = "✔" if fname in self.done else "—"
        self.root.title(f"YOLO 标注修正器 v2.2 - {fname} [{self.idx+1}/{len(self.files)}]")
        self.status.config(
            text=f"[{self.idx+1}/{len(self.files)}] {fname}  完成标记:{mark}  "
                 f"框数:{len(self.boxes)}  {'●未保存' if self.dirty else '已保存'}  "
                 f"当前类别:{self.cur_class} {CLASSES[self.cur_class]}  "
                 f"| 已完成 {len(self.done)}/{len(self.files)}")

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, (cid, *_ ) in enumerate(self.boxes):
            self.listbox.insert(tk.END, f"{i}: {CLASSES[cid]}")
        if self.selected is not None:
            self.listbox.selection_set(self.selected)

    # ---------- 文件列表 / 工具栏 ----------
    def _update_toolbar(self):
        self.save_lbl.config(text=f"标签保存到: {self.lbl_dir}")

    def _refresh_files(self):
        """刷新右侧文件列表：✔=已保存，并高亮当前图"""
        self._file_lock = True
        self.file_list.delete(0, tk.END)
        for f in self.files:
            self.file_list.insert(tk.END, ("✔ " if f in self.done else "   ") + f)
        self.file_list.selection_clear(0, tk.END)
        self.file_list.selection_set(self.idx)
        self.file_list.see(self.idx)
        self._file_lock = False

    def on_file_select(self, e):
        if self._file_lock:
            return
        sel = self.file_list.curselection()
        if sel and sel[0] != self.idx:
            self._switch(sel[0])

    # ---------- 打开/切换文件夹（LabelImg 的 Open Dir / Change Save Dir） ----------
    def open_folder(self):
        init = self.progress.get("config", {}).get("last_dir") or self.img_dir
        d = filedialog.askdirectory(initialdir=init, title="选择图片文件夹")
        if not d:
            return
        if self.dirty:
            self.save()                      # 改动先按原目录落盘
        start = self._use_folder(d)
        if start is None:
            messagebox.showwarning("提示", f"文件夹不存在或里面没有图片：\n{d}")
            return
        self.selected = None
        self._update_toolbar()
        self.load_image(start)

    def change_save_dir(self):
        d = filedialog.askdirectory(initialdir=self.lbl_dir, title="选择标签保存文件夹")
        if not d:
            return
        if self.dirty:
            self.save()                      # 改动先按原目录落盘
        self.lbl_dir = os.path.normpath(d)
        self._update_toolbar()
        # 从新目录重新读入当前图标签，避免"看到的不等于存进去的"
        self.boxes = self.read_label(os.path.splitext(self.files[self.idx])[0])
        self.selected = None
        self.dirty = False
        self.redraw()

    def next_unlabeled(self):
        """跳到下一张还没有标签、也没标记过完成的图"""
        n = len(self.files)
        for step in range(1, n + 1):
            j = (self.idx + step) % n
            f = self.files[j]
            if f in self.done:
                continue
            lp = os.path.join(self.lbl_dir, os.path.splitext(f)[0] + ".txt")
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                continue
            if j != self.idx:
                self._switch(j)
            return
        self.status.config(text="全部图片都已标注/检查过了 ✔")

    # ---------- 坐标换算 ----------
    def to_norm(self, x1, y1, x2, y2):
        x1 = max(self.ox, min(x1, self.ox + self.disp_w))
        x2 = max(self.ox, min(x2, self.ox + self.disp_w))
        y1 = max(self.oy, min(y1, self.oy + self.disp_h))
        y2 = max(self.oy, min(y2, self.oy + self.disp_h))
        cx = ((x1 + x2) / 2 - self.ox) / self.disp_w
        cy = ((y1 + y2) / 2 - self.oy) / self.disp_h
        return cx, cy, abs(x2 - x1) / self.disp_w, abs(y2 - y1) / self.disp_h

    def hit_box(self, x, y):
        for i, (cid, cx, cy, w, h) in enumerate(self.boxes):
            bx1 = self.ox + (cx - w / 2) * self.disp_w
            by1 = self.oy + (cy - h / 2) * self.disp_h
            bx2 = self.ox + (cx + w / 2) * self.disp_w
            by2 = self.oy + (cy + h / 2) * self.disp_h
            if bx1 - 4 <= x <= bx2 + 4 and by1 - 4 <= y <= by2 + 4:
                return i
        return None

    # ---------- 事件 ----------
    def on_press(self, e):
        self.drag_start = (e.x, e.y)
        self.rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                    outline="white", dash=(3, 2))

    def on_drag(self, e):
        if self.drag_start:
            self.canvas.coords(self.rect_id, *self.drag_start, e.x, e.y)

    def on_release(self, e):
        if not self.drag_start:
            return
        x1, y1 = self.drag_start
        x2, y2 = e.x, e.y
        self.canvas.delete(self.rect_id)
        self.drag_start = None
        if abs(x2 - x1) < 8 or abs(y2 - y1) < 8:      # 小拖拽视为点击选中
            i = self.hit_box(x1, y1)
            self.selected = i
            self.redraw()
            return
        cx, cy, w, h = self.to_norm(x1, y1, x2, y2)
        if w > 0.005 and h > 0.005:
            self.boxes.append((self.cur_class, cx, cy, w, h))
            self.selected = len(self.boxes) - 1
            self.dirty = True
            self.redraw()
            self._maybe_autosave()

    def on_double_click(self, e):
        i = self.hit_box(e.x, e.y)
        if i is not None:
            self.boxes.pop(i)
            self.selected = None
            self.dirty = True
            self.redraw()
            self._maybe_autosave()

    def on_list_select(self, e):
        sel = self.listbox.curselection()
        if sel:
            self.selected = sel[0]
            self.redraw()

    def on_key(self, e):
        k = e.keysym
        if k == "Tab":
            self.cycle_select()
            return "break"                      # 阻止焦点跳走
        if k in ("1", "2", "3"):
            self.set_class(int(k) - 1)
        elif k.lower() == "a":
            self.prev()
        elif k.lower() == "d":
            self.next()
        elif k.lower() == "f":
            self.next_unlabeled()
        elif k == "Delete":
            self.delete_selected()
        elif k.lower() == "s" and (e.state & 0x4):
            self.save()

    def cycle_select(self):
        """Tab：在本图已标注框之间循环选中"""
        if not self.boxes:
            return
        self.selected = 0 if self.selected is None else (self.selected + 1) % len(self.boxes)
        self.redraw()

    def _maybe_autosave(self):
        if self.autosave_var.get():
            self.save()

    def delete_selected(self):
        if self.selected is not None:
            self.boxes.pop(self.selected)
            self.selected = None
            self.dirty = True
            self.redraw()
            self._maybe_autosave()

    # ---------- 保存/翻页 ----------
    def save(self):
        os.makedirs(self.lbl_dir, exist_ok=True)
        stem = os.path.splitext(self.files[self.idx])[0]
        with open(os.path.join(self.lbl_dir, stem + ".txt"), "w") as fp:
            for cid, cx, cy, w, h in self.boxes:
                fp.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        cls_txt = os.path.join(self.lbl_dir, "classes.txt")
        if not os.path.exists(cls_txt):
            with open(cls_txt, "w", encoding="utf-8") as cf:
                cf.write("\n".join(CLASSES) + "\n")
        self.done.add(self.files[self.idx])
        self.dirty = False
        self.redraw()
        self._refresh_files()
        self._persist()

    def _switch(self, new_idx):
        if self.dirty:
            self.save()
        self.load_image(new_idx)

    def prev(self):
        self._switch(self.idx - 1)

    def next(self):
        self._switch(self.idx + 1)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="YOLO 标注修正器 v2.2")
    ap.add_argument("--split", default="val", choices=["train", "val"],
                    help="数据集子集（默认 val；数据集标注与 --dir 二选一）")
    ap.add_argument("--dir", default=None,
                    help="图片文件夹（默认 yolo_project/dataset/images/<split>），标签存到其下 labels 或同名 labels 目录")
    ap.add_argument("--labels", default=None,
                    help="标签保存文件夹（不填则按图片目录自动推算）")
    args = ap.parse_args()
    Annotator(args.split, img_dir=args.dir, lbl_dir=args.labels).run()


if __name__ == "__main__":
    main()
