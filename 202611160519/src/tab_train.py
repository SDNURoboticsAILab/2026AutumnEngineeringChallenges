# tab_train.py
# 训练页 Tab（v0.8.0）
# 新增：
#   - 🔍 找模型 按钮
#   - 训练前检查重复框并提示
import json, os, shutil, subprocess, sys, threading, time, traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import psutil
from ultralytics import YOLO

from core import TextRedirector, IMG_EXTS, dedupe_boxes, CLASS_NAMES


class TrainTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._stop_request_time = None
        self._batch_start_time = None
        self._avg_batch_time = None
        self._stop_display_id = None
        self._current_output_dir = None
        self._build_ui()

    def _build_ui(self):
        dev = ttk.LabelFrame(self, text="设备状态")
        dev.pack(fill=tk.X, padx=10, pady=(8, 4))
        self.device_labels = {}
        for i, (name, key) in enumerate([("CPU","cpu"),("内存","mem"),("GPU","gpu"),("显存","vram")]):
            ttk.Label(dev, text=f"{name}:", width=6).grid(row=0, column=i*2, sticky=tk.W, padx=(8,2), pady=4)
            lbl = ttk.Label(dev, text="--", width=22, foreground="#007acc")
            lbl.grid(row=0, column=i*2+1, sticky=tk.W, padx=(0,8))
            self.device_labels[key] = lbl

        prog = ttk.LabelFrame(self, text="训练进度")
        prog.pack(fill=tk.X, padx=10, pady=4)
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(prog, variable=self.progress_var, maximum=100, length=400).pack(fill=tk.X, padx=8, pady=(6,2))
        info = ttk.Frame(prog); info.pack(fill=tk.X, padx=8, pady=2)
        self.epoch_var = tk.StringVar(value="Epoch: -/-")
        self.metric_var = tk.StringVar(value="等待训练...")
        self.eta_var = tk.StringVar(value="已用: -  预计剩余: -")
        ttk.Label(info, textvariable=self.epoch_var, width=35).pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.metric_var, width=50).pack(side=tk.LEFT, padx=10)
        ttk.Label(info, textvariable=self.eta_var, width=30).pack(side=tk.LEFT)
        self.stop_status_var = tk.StringVar(value="")
        self.stop_status_lbl = ttk.Label(prog, textvariable=self.stop_status_var,
                                          foreground="#cc6600", font=('Consolas', 10, 'bold'))

        f = ttk.LabelFrame(self, text="训练参数"); f.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(f, text="data.yaml:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(f, textvariable=self.app.data_yaml, width=70).grid(row=0, column=1, padx=5)
        ttk.Button(f, text="选择", command=self._choose_yaml).grid(row=0, column=2, padx=5)
        ttk.Label(f, text="基础权重:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.weight_combo = ttk.Combobox(f, textvariable=self.app.weights_train, values=[], width=25, state="readonly")
        self.weight_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(f, text="重新扫描", command=self._rescan).grid(row=1, column=2, padx=5)
        ttk.Label(f, text="继续训练:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(f, textvariable=self.app.continue_from, width=60).grid(row=2, column=1, padx=5)
        ttk.Button(f, text="选择best.pt", command=self._choose_continue).grid(row=2, column=2, padx=5)
        row = ttk.Frame(f); row.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        for lbl, var, w in [("epochs:", self.app.epochs, 6), ("imgsz:", self.app.imgsz, 6), ("batch:", self.app.batch, 6)]:
            ttk.Label(row, text=lbl).pack(side=tk.LEFT, padx=(8,0))
            ttk.Entry(row, textvariable=var, width=w).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="device:").pack(side=tk.LEFT, padx=(8,0))
        ttk.Combobox(row, textvariable=self.app.device, values=["cpu","0","1"], width=5, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Label(f, text="输出目录:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(f, textvariable=self.app.project, width=70).grid(row=4, column=1, padx=5)
        ttk.Button(f, text="选择", command=self._choose_project).grid(row=4, column=2, padx=5)
        ttk.Label(f, text="版本名:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(f, textvariable=self.app.exp_name, width=70).grid(row=5, column=1, padx=5)

        btns = ttk.Frame(self); btns.pack(fill=tk.X, padx=10, pady=4)
        self.btn_start = ttk.Button(btns, text="开始训练", command=self.start_training)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_stop = ttk.Button(btns, text="停止训练", command=self.stop_training, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="打开输出目录", command=self._open_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="数据集统计", command=self.show_stats).pack(side=tk.LEFT, padx=5)
        # ===== 🔍 找模型 =====
        ttk.Button(btns, text="🔍 找模型", command=self._open_finder).pack(side=tk.LEFT, padx=5)

        hist = ttk.LabelFrame(self, text="版本历史"); hist.pack(fill=tk.X, padx=10, pady=4)
        self.hist_text = tk.Text(hist, height=5, font=('Consolas', 9))
        self.hist_text.pack(fill=tk.X, padx=5, pady=5)
        self.hist_text.config(state=tk.DISABLED)

        logf = ttk.LabelFrame(self, text="训练日志"); logf.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        sb = ttk.Scrollbar(logf); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log = tk.Text(logf, wrap=tk.WORD, yscrollcommand=sb.set, font=('Consolas', 9),
                           bg='#1e1e1e', fg='#d4d4d4')
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb.config(command=self.log.yview)
        self.log.config(state=tk.DISABLED)

        self.app._monitor_running = True
        self._refresh_device()

    # ==================== 设备监控 ====================
    def _get_gpu(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=1,
                creationflags=subprocess.CREATE_NO_WINDOW).decode().strip()
            if not out: return None
            g, mu, mt = [x.strip() for x in out.splitlines()[0].split(",")]
            return {"gpu_util": int(g), "vram_used": int(mu), "vram_total": int(mt)}
        except Exception: return None

    def _refresh_device(self):
        if not self.app._monitor_running: return
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self.device_labels["cpu"].config(text=f"{cpu:.1f}%")
            self.device_labels["mem"].config(text=f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent:.0f}%)")
            gpu = self._get_gpu()
            if gpu:
                self.device_labels["gpu"].config(text=f"{gpu['gpu_util']}%")
                self.device_labels["vram"].config(
                    text=f"{gpu['vram_used']}/{gpu['vram_total']} MB "
                         f"({gpu['vram_used']*100//max(gpu['vram_total'],1)}%)")
            else:
                self.device_labels["gpu"].config(text="不可用")
                self.device_labels["vram"].config(text="不可用")
        except Exception: pass
        self.after(1000, self._refresh_device)

    # ==================== 路径选择 ====================
    def _choose_yaml(self):
        p = filedialog.askopenfilename(filetypes=[("YAML","*.yaml *.yml"),("All","*.*")])
        if p: self.app.data_yaml.set(p)
    def _choose_project(self):
        p = filedialog.askdirectory()
        if p: self.app.project.set(p)
    def _choose_continue(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch","*.pt"),("All","*.*")])
        if p: self.app.continue_from.set(p)
    def _rescan(self):
        self.app._scan_project_files()
        self.app._autofill_paths()
    def _open_output(self):
        p = Path(self.app.project.get()) / self.app.exp_name.get()
        if p.exists(): os.startfile(str(p))
        else: messagebox.showinfo("提示", f"目录不存在:\n{p}")

    def _open_finder(self):
        try:
            from model_finder import open_model_finder
            open_model_finder(self, self.app)
        except Exception as e:
            messagebox.showerror("打开失败", f"{type(e).__name__}: {e}")

    # ==================== 版本历史 ====================
    def refresh_history(self):
        self.hist_text.config(state=tk.NORMAL)
        self.hist_text.delete("1.0", tk.END)
        vs = self.app.history.get("versions", [])
        if not vs: self.hist_text.insert(tk.END, "暂无版本记录\n")
        else:
            for v in vs:
                self.hist_text.insert(tk.END,
                    f"{v['name']:<16} epochs={v['epochs']:<4} mAP50={v['mAP50']:.3f} {v['time']}\n")
        self.hist_text.config(state=tk.DISABLED)

    # ==================== 停止显示 ====================
    def _start_stop_display(self):
        try:
            self.stop_status_lbl.pack(fill=tk.X, padx=8, pady=(4, 2))
        except Exception: pass
        self._update_stop_display()

    def _stop_stop_display(self):
        if self._stop_display_id is not None:
            try: self.after_cancel(self._stop_display_id)
            except Exception: pass
            self._stop_display_id = None
        try: self.stop_status_lbl.pack_forget()
        except Exception: pass
        self.stop_status_var.set("")

    def _update_stop_display(self):
        if not self.app.is_training or not self.app.stop_flag:
            self._stop_stop_display(); return
        if self._stop_request_time is None:
            self._stop_display_id = self.after(500, self._update_stop_display); return
        waited = time.time() - self._stop_request_time
        cur_batch_elapsed = time.time() - self._batch_start_time if self._batch_start_time else 0
        self.stop_status_var.set(
            f"正在停止...  已等待 {waited:.0f}s  "
            f"|  当前 batch 已执行 {cur_batch_elapsed:.0f}s  "
            f"|  即将强制中断")
        self._stop_display_id = self.after(500, self._update_stop_display)

    # ==================== 训练前重复框检查 ====================
    def _check_duplicate_gt(self):
        """
        检查 dataset/labels/train 里的标签是否有重复框。
        返回 (n_images_with_dup, total_dup, list_examples)
        """
        img_dir = Path(self.app.auto_img_dir.get())
        lbl_dir = Path(self.app.auto_labels_dir.get())
        if not lbl_dir.exists():
            return 0, 0, []
        dup_imgs = 0
        total_dup = 0
        examples = []
        for lbl in lbl_dir.glob("*.txt"):
            boxes = []
            try:
                with open(lbl, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        # 转 xyxy（用 0-1 归一化坐标即可）
                        boxes.append({
                            "cls_id": cid,
                            "cls_name": CLASS_NAMES.get(cid, str(cid)),
                            "xyxy": [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                        })
            except Exception:
                continue
            if len(boxes) < 2:
                continue
            _keep, rejected = dedupe_boxes(boxes, iou_thr=0.55, contain_thr=0.80)
            if rejected:
                dup_imgs += 1
                total_dup += len(rejected)
                if len(examples) < 8:
                    examples.append((lbl.name, len(rejected), len(boxes)))
        return dup_imgs, total_dup, examples

    # ==================== 开始训练 ====================
    def start_training(self):
        app = self.app
        if app.is_training: return
        if not Path(app.data_yaml.get()).exists():
            messagebox.showerror("错误", "data.yaml 不存在"); return
        try:
            epochs = int(app.epochs.get()); imgsz = int(app.imgsz.get()); batch = int(app.batch.get())
        except Exception:
            messagebox.showerror("错误", "参数必须是整数"); return
        cf = app.continue_from.get().strip()
        if cf and not Path(cf).exists():
            app.continue_from.set("")
            messagebox.showwarning("提示", "继续训练权重不存在，已回退到基础权重")
        base = app.continue_from.get().strip() or app.weights_train.get()
        if not Path(base).exists():
            messagebox.showerror("错误", f"初始权重不存在:\n{base}"); return
        if not app.exp_name.get() or app.exp_name.get() == "exp":
            app.exp_name.set(f"exp_v{len(app.history.get('versions', [])) + 1}")

        # ===== 训练前检查重复框 =====
        dup_imgs, total_dup, examples = self._check_duplicate_gt()
        if dup_imgs > 0:
            lines = [f"在 {dup_imgs} 张图片里发现 {total_dup} 个疑似重复框。", ""]
            for name, n, total in examples[:8]:
                lines.append(f"  · {name}  重复 {n} / 共 {total}")
            if len(examples) > 8:
                lines.append(f"  ... 还有 {dup_imgs - 8} 张")
            lines.append("")
            lines.append("建议：先到「GT 库管理」页点「🧹 去重」，或手动合并，再训练。")
            lines.append("")
            lines.append("是否仍然继续训练？")
            if not messagebox.askyesno("⚠ 发现疑似重复框", "\n".join(lines)):
                return

        app.stop_flag = False
        app.is_training = True
        self._stop_request_time = None
        self._batch_start_time = None
        self._avg_batch_time = None
        self._stop_stop_display()

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        app.status.set("训练中...")
        self.progress_var.set(0)
        self.epoch_var.set("准备中...")
        self.metric_var.set("等待第一个 batch...")
        self.log.config(state=tk.NORMAL); self.log.delete('1.0', tk.END); self.log.config(state=tk.DISABLED)
        threading.Thread(target=self._do_train, args=(epochs, imgsz, batch), daemon=True).start()

    def stop_training(self):
        if not self.app.is_training: return
        self.app.stop_flag = True
        self._stop_request_time = time.time()
        self.btn_stop.config(state=tk.DISABLED)
        self.app.status.set("停止请求已发送")
        self._start_stop_display()

    def _do_train(self, epochs, imgsz, batch):
        app = self.app
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = TextRedirector(self.log); sys.stderr = sys.stdout
        app.train_start_time = time.time()
        app.total_epochs = epochs; app.current_epoch = 0
        base = app.continue_from.get().strip() or app.weights_train.get()
        is_continue = bool(app.continue_from.get().strip())
        output_dir = Path(app.project.get()) / app.exp_name.get()
        self._current_output_dir = output_dir

        class _StopTraining(Exception):
            pass

        try:
            model = YOLO(base)

            def on_batch_start(trainer):
                self._batch_start_time = time.time()
                if app.stop_flag:
                    raise _StopTraining("stop_requested")

            def on_batch_end(trainer):
                try:
                    if self._batch_start_time is not None:
                        dur = time.time() - self._batch_start_time
                        if self._avg_batch_time is None:
                            self._avg_batch_time = dur
                        else:
                            self._avg_batch_time = self._avg_batch_time * 0.7 + dur * 0.3
                    if app.stop_flag:
                        raise _StopTraining("stop_requested")
                    ce = trainer.epoch; te = trainer.epochs
                    try: bpe = len(trainer.train_loader)
                    except Exception: bpe = 1
                    cb = getattr(trainer, "batch_idx", 0)
                    done = ce*bpe + cb + 1; total = te*bpe
                    pct = done/total*100 if total else 0
                    bl = cl = 0.0
                    try:
                        tl = trainer.tloss
                        if tl is not None and len(tl) >= 2: bl = float(tl[0]); cl = float(tl[1])
                    except Exception: pass
                    el = time.time() - app.train_start_time
                    per = el/done if done > 0 else 0; rem = per*(total-done)
                    def ui():
                        self.progress_var.set(pct)
                        self.epoch_var.set(f"Epoch: {ce+1}/{te}  Batch: {cb+1}/{bpe}")
                        self.metric_var.set(f"box_loss: {bl:.4f}  cls_loss: {cl:.4f}")
                        self.eta_var.set(f"已用: {self._fmt(el)}  剩余: {self._fmt(rem)}")
                    self.after(0, ui)
                except _StopTraining:
                    raise
                except Exception as e:
                    print(f"[进度回调异常] {e}")

            model.add_callback("on_train_batch_start", on_batch_start)
            model.add_callback("on_train_batch_end", on_batch_end)

            def on_epoch_end(trainer):
                try:
                    if app.stop_flag:
                        raise _StopTraining("stop_requested")
                    m = getattr(trainer, "metrics", None)
                    mp = float(m.get("metrics/mAP50(B)", 0.0)) if m else 0
                    pc = float(m.get("metrics/precision(B)", 0.0)) if m else 0
                    rc = float(m.get("metrics/recall(B)", 0.0)) if m else 0
                    c = trainer.epoch + 1; t = trainer.epochs
                    def ui():
                        self.metric_var.set(f"Epoch {c}/{t}  mAP50:{mp:.3f}  P:{pc:.3f}  R:{rc:.3f}")
                    self.after(0, ui)
                except _StopTraining:
                    raise
                except Exception: pass
            model.add_callback("on_train_epoch_end", on_epoch_end)

            aug_n = len(list(Path(app.auto_img_dir.get()).glob("aug_*")))
            use_mosaic = 0.5 if aug_n > 100 else 1.0
            lr0 = 0.001 if is_continue else 0.01

            interrupted = False
            try:
                model.train(data=app.data_yaml.get(), epochs=epochs, imgsz=imgsz, batch=batch,
                            device=app.device.get(), project=app.project.get(), name=app.exp_name.get(),
                            exist_ok=True, lr0=lr0, lrf=0.01, cls=1.5,
                            degrees=10.0, translate=0.15, scale=0.5, shear=2.0,
                            perspective=0.0005, fliplr=0.5, flipud=0.0,
                            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                            mosaic=use_mosaic, mixup=0.05, copy_paste=0.0, patience=15)
            except _StopTraining:
                interrupted = True
                print("\n[停止] 用户请求停止，已强制中断训练。\n")
            except Exception as e:
                if app.stop_flag or "stop_requested" in str(e).lower():
                    interrupted = True
                    print("\n[停止] 训练已被中断。\n")
                else:
                    raise

            if interrupted:
                try:
                    if output_dir.exists():
                        shutil.rmtree(output_dir, ignore_errors=True)
                        print(f"[取消] 已删除未完成的版本目录: {output_dir}")
                except Exception as e:
                    print(f"[取消失败] {e}")
                def notify():
                    messagebox.showinfo("训练已停止",
                        f"已强制中断训练\n\n已删除当前未完成的版本目录:\n{output_dir}")
                self.after(0, notify)
                return

            best = Path(app.project.get()) / app.exp_name.get() / "weights" / "best.pt"
            mAP50 = 0.0
            csv = Path(app.project.get()) / app.exp_name.get() / "results.csv"
            if csv.exists():
                try:
                    with open(csv, "r", encoding="utf-8") as f: lines = f.readlines()
                    if len(lines) > 1:
                        hdrs = [h.strip() for h in lines[0].split(",")]; last = lines[-1].split(",")
                        for i, h in enumerate(hdrs):
                            if "mAP50" in h and "95" not in h: mAP50 = float(last[i]); break
                except Exception: pass
            app.history.setdefault("versions", []).append({
                "name": app.exp_name.get(),
                "time": datetime.now().isoformat(timespec="seconds"),
                "epochs": app.current_epoch, "mAP50": mAP50})
            app._save_history()
            self.after(0, self.refresh_history)
            self.after(0, app._scan_project_files)
            self.after(0, app._autofill_paths)
            self.after(0, lambda: messagebox.showinfo("训练完成",
                f"最优权重:\n{best}\n\nmAP50: {mAP50:.3f}\n学习率: {lr0}\nMosaic: {use_mosaic}"))

        except Exception as e:
            print("\n===== 训练异常 =====\n" + traceback.format_exc())
            self.after(0, lambda: messagebox.showerror("训练出错", f"{type(e).__name__}: {e}"))
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            app.is_training = False
            app.stop_flag = False
            self._stop_request_time = None
            self._batch_start_time = None
            self.after(0, self._stop_stop_display)
            self.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))
            self.after(0, lambda: app.status.set("就绪"))

    @staticmethod
    def _fmt(s):
        s = int(s); h = s//3600; m = (s%3600)//60; sec = s%60
        if h > 0: return f"{h}h{m}m"
        if m > 0: return f"{m}m{sec}s"
        return f"{sec}s"

    # ==================== 数据集统计 ====================
    def show_stats(self):
        try:
            app = self.app
            img_dir = Path(app.auto_img_dir.get()); lbl_dir = Path(app.auto_labels_dir.get())
            if not img_dir.exists(): messagebox.showerror("错误", "图片目录不存在"); return
            all_imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            aug_imgs = [p for p in all_imgs if p.name.startswith("aug_")]
            origin_imgs = [p for p in all_imgs if not p.name.startswith("aug_")]
            cls_counts = {0: 0, 1: 0, 2: 0}; n_empty = n_with = 0
            for img in all_imgs:
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists(): continue
                with open(lbl, "r", encoding="utf-8") as f:
                    lines = [l for l in f if l.strip()]
                if not lines: n_empty += 1; continue
                n_with += 1
                for line in lines:
                    try:
                        cid = int(line.split()[0])
                        if cid in cls_counts: cls_counts[cid] += 1
                    except Exception: pass
            val_dir = img_dir.parent / "val"
            n_val = len([p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]) if val_dir.exists() else 0
            ratio = len(aug_imgs) / max(len(origin_imgs), 1)

            # 重复框统计
            dup_imgs, total_dup, _examples = self._check_duplicate_gt()

            win = tk.Toplevel(self); win.title("数据集统计"); win.geometry("660x600")
            win.resizable(True, True)
            win.bind("<Escape>", lambda e: win.destroy())
            txt = tk.Text(win, font=('Consolas', 11), bg='#f8f8f8', wrap=tk.WORD)
            txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            lines = ["="*50, "                 数据集统计", "="*50, "",
                     "【训练集】", f"  总图片: {len(all_imgs)}",
                     f"  · 原图: {len(origin_imgs)}",
                     f"  · 增强 (aug_): {len(aug_imgs)}",
                     f"  增强比例: {ratio:.1f}x", "",
                     "【标注】", f"  有标注: {n_with}", f"  空标注(负样本): {n_empty}", "",
                     "【类别分布】",
                     f"  obstacle: {cls_counts[0]} 个框",
                     f"  cola: {cls_counts[1]} 个框",
                     f"  football: {cls_counts[2]} 个框", "",
                     "【验证集】", f"  图片数: {n_val}", "",
                     "【重复框检查】",
                     f"  疑似重复框的图片: {dup_imgs} 张",
                     f"  重复框总数: {total_dup} 个", "",
                     "="*50]
            warns = []
            if len(aug_imgs) == 0: warns.append("没有增强图")
            if cls_counts[1] < 30: warns.append(f"cola 只有 {cls_counts[1]}，建议 >50")
            if cls_counts[2] < 30: warns.append(f"football 只有 {cls_counts[2]}，建议 >50")
            if cls_counts[0] < 50: warns.append(f"obstacle 只有 {cls_counts[0]}，建议 >100")
            if n_empty < 5: warns.append(f"空标注只有 {n_empty} 张，建议加 5~10 张纯背景")
            if len(aug_imgs) > 0 and ratio < 2: warns.append(f"增强比例 {ratio:.1f}x，建议 3~5x")
            if dup_imgs > 0: warns.append(f"有 {dup_imgs} 张图存在重复框，建议先「🧹 去重」")
            if warns:
                lines.append(""); lines.append("【健康度提示】")
                for w in warns: lines.append(f"  {w}")
            txt.insert(tk.END, "\n".join(lines))
            txt.config(state=tk.DISABLED)
            ttk.Button(win, text="关闭", command=win.destroy).pack(pady=8)
        except Exception as e:
            messagebox.showerror("统计失败", f"{e}")