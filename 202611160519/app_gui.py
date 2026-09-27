# app_gui.py
# 主入口：串起所有 Tab（v1.3）
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import json, os, shutil, threading, time, traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk

from core import (ErrorBook, GroundTruth, ManualMergeLog, GuidanceLog,
                  PROJECT_ROOT)
from learning_rules import LearningRules
from tab_train import TrainTab
from tab_auto import AutoTab
from tab_gt import GTTab
from tab_aug import AugTab
from tab_review import ReviewTab
from tab_error import ErrorTab


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("明子翔的YOLO项目")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 720)

        cwd = PROJECT_ROOT

        # 全局状态
        self.detector = None
        self.is_training = False
        self.stop_flag = False
        self.train_start_time = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self._monitor_running = False

        self._ft_lock = threading.Lock()
        self.finetune_pending = 0
        self.finetune_batch_size = 20
        self.finetune_epochs = 5

        # 全局变量
        self.data_yaml = tk.StringVar(value=str(cwd / "data.yaml"))
        self.weights_train = tk.StringVar(value="yolo11n.pt")
        self.epochs = tk.IntVar(value=100)
        self.imgsz = tk.IntVar(value=640)
        self.batch = tk.IntVar(value=16)
        self.device = tk.StringVar(value="cpu")
        self.project = tk.StringVar(value=str(cwd / "runs" / "train"))
        self.exp_name = tk.StringVar(value="exp")
        self.continue_from = tk.StringVar(value="")
        self.weights_pred = tk.StringVar(
            value=str(cwd / "runs" / "train" / "exp" / "weights" / "best.pt"))
        self.conf_thr = tk.DoubleVar(value=0.30)

        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if not has_cuda:
            self.imgsz.set(416)
            self.batch.set(8)

        self.auto_weights = tk.StringVar(value="yolo11n.pt")
        self.auto_img_dir = tk.StringVar(value=str(cwd / "dataset" / "images" / "train"))
        self.auto_pending_dir = tk.StringVar(value=str(cwd / "dataset" / "labels_pending"))
        self.auto_labels_dir = tk.StringVar(value=str(cwd / "dataset" / "labels" / "train"))

        # 数据
        self.error_book = ErrorBook(cwd / "results" / "error_book.json")
        self.ground_truth = GroundTruth(cwd / "results" / "ground_truth")
        self.merge_log = ManualMergeLog(cwd / "results" / "manual_merge.json")
        self.guidance = GuidanceLog(cwd / "results" / "guidance.json")
        self.learning_rules = LearningRules(cwd / "results" / "learning_rules.json")

        self.available_weights = []
        self.available_yamls = []
        self.history = {"versions": []}
        self._scan_project_files()
        self._load_history()

        self.status = tk.StringVar(value="就绪")
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.root.after(500, self._auto_load_best)

    def _scan_project_files(self):
        cwd = PROJECT_ROOT
        weights = [p for p in cwd.rglob("*.pt")
                   if not ("runs" in p.parts and "weights" not in p.parts)]
        weights.sort(key=lambda p: (0 if p.parent == cwd else (1 if "weights" in p.parts else 2)))
        self.available_weights = weights
        yamls = list(cwd.rglob("data.yaml"))
        yamls.sort(key=lambda p: (len(p.parts), str(p)))
        self.available_yamls = yamls

    def _autofill_paths(self):
        if self.available_yamls:
            self.data_yaml.set(str(self.available_yamls[0]))
        names = [p.name for p in self.available_weights]
        if names:
            if hasattr(self, "train_tab") and hasattr(self.train_tab, "weight_combo"):
                self.train_tab.weight_combo['values'] = names
            if hasattr(self, "auto_tab") and hasattr(self.auto_tab, "auto_weight_combo"):
                self.auto_tab.auto_weight_combo['values'] = names
            if hasattr(self, "gt_tab") and hasattr(self.gt_tab, "pred_combo"):
                self.gt_tab.pred_combo['values'] = names
            best = next((p for p in self.available_weights if p.name == "best.pt"), None)
            if best: self.weights_pred.set(str(best))
            elif self.available_weights: self.weights_pred.set(str(self.available_weights[0]))
        self.root.after(200, lambda: self.status.set(
            f"扫描到 {len(self.available_weights)} 个权重, "
            f"{len(self.available_yamls)} 个 data.yaml"))

    def _history_path(self):
        return Path(self.project.get()) / "history.json"

    def _load_history(self):
        p = self._history_path()
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = {"versions": []}

    def _save_history(self):
        p = self._history_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def _auto_load_best(self):
        p = self.weights_pred.get()
        if p and Path(p).exists():
            try:
                from predict import Detector
                self.detector = Detector(p)
                self.status.set(f"已自动加载: {Path(p).name}")
            except Exception as e:
                self.status.set(f"自动加载失败: {e}")
        if hasattr(self, "gt_tab"):
            self.gt_tab.refresh_model_info()

    def refresh_all_gt_views(self):
        try:
            if hasattr(self, "gt_tab") and self.gt_tab is not None:
                self.gt_tab.refresh_gt_list()
                self.gt_tab.refresh_gt_stat()
        except Exception as e:
            print(f"[刷新 GT 管理页失败] {e}")
        try:
            if hasattr(self, "review_tab") and self.review_tab is not None:
                self.review_tab.refresh_gt_list()
        except Exception as e:
            print(f"[刷新开卷学习页失败] {e}")

    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.columnconfigure(0, weight=1)

        nb = ttk.Notebook(self.root)
        nb.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))
        self.notebook = nb

        self.train_tab = TrainTab(nb, self); nb.add(self.train_tab, text="训练")
        self.auto_tab = AutoTab(nb, self); nb.add(self.auto_tab, text="自动标注")
        self.gt_tab = GTTab(nb, self); nb.add(self.gt_tab, text="GT 库管理")
        self.aug_tab = AugTab(nb, self); nb.add(self.aug_tab, text="数据增强")
        self.review_tab = ReviewTab(nb, self); nb.add(self.review_tab, text="开卷学习")
        self.error_tab = ErrorTab(nb, self); nb.add(self.error_tab, text="错题本")

        self._autofill_paths()
        self.train_tab.refresh_history()
        self.error_tab.refresh()
        self.review_tab.refresh_gt_list()
        self.gt_tab.refresh_gt_list()
        self.gt_tab.refresh_gt_stat()

        bottom = ttk.Frame(self.root, height=32)
        bottom.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))
        bottom.grid_propagate(False)
        ttk.Label(bottom, textvariable=self.status, relief=tk.SUNKEN,
                  anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        ttk.Button(bottom, text="退出", command=self.on_exit).pack(side=tk.RIGHT)

    def _do_batch_finetune(self):
        try:
            best_pt = self.weights_pred.get()
            if not Path(best_pt).exists():
                print("[批处理微调] best.pt 不存在"); return
            self.root.after(0, lambda: self.status.set("后台批处理微调..."))
            from ultralytics import YOLO
            ts = datetime.now().strftime("%m%d_%H%M%S")
            m = YOLO(best_pt)
            aug_n = len(list(Path(self.auto_img_dir.get()).glob("aug_*")))
            use_mosaic = 0.5 if aug_n > 100 else 1.0
            m.train(data=self.data_yaml.get(),
                    epochs=self.finetune_epochs,
                    imgsz=int(self.imgsz.get()),
                    batch=int(self.batch.get()),
                    device=self.device.get(),
                    project=self.project.get(),
                    name=f"batch_ft_{ts}", exist_ok=True,
                    lr0=0.001, lrf=0.01, cls=1.5,
                    degrees=10.0, translate=0.15, scale=0.5, fliplr=0.5,
                    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                    mosaic=use_mosaic, mixup=0.05, patience=15, verbose=False)
            new_best = Path(self.project.get()) / f"batch_ft_{ts}" / "weights" / "best.pt"
            if new_best.exists():
                shutil.copy2(new_best, best_pt)
                self.root.after(0, self._reload_detector)
                self.root.after(0, lambda: self.status.set(f"批处理微调完成 ({ts})"))
            else:
                self.root.after(0, lambda: self.status.set("微调未生成 best.pt"))
        except Exception as e:
            print(f"[批处理微调失败] {traceback.format_exc()}")
            self.root.after(0, lambda: self.status.set(f"微调失败: {e}"))

    def _reload_detector(self):
        try:
            from predict import Detector
            self.detector = Detector(self.weights_pred.get())
            if hasattr(self, "gt_tab"):
                self.gt_tab.refresh_model_info()
        except Exception as e:
            print(f"[重载模型失败] {e}")

    def _backup_best_pt(self):
        try:
            runs_dir = PROJECT_ROOT / "runs"
            bfs = list(runs_dir.rglob("best.pt")) if runs_dir.exists() else []
            if not bfs: return
            bd = PROJECT_ROOT / "results" / "best_backups"
            bd.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            for b in bfs:
                try:
                    en = b.parent.parent.name
                    dst = bd / f"{en}_{ts}_best.pt"
                    shutil.copy2(b, dst)
                    print(f"[备份] {en} -> {dst.name}")
                except Exception as e:
                    print(f"[备份失败] {e}")
        except Exception as e:
            print(f"[备份异常] {e}")

    def on_exit(self):
        if self.is_training:
            if not messagebox.askyesno("确认退出", "训练中。强制退出？（best.pt 保留上一轮）"):
                return
            self.stop_flag = True
            self.status.set("正在停止训练...")
            t0 = time.time()
            while self.is_training and (time.time() - t0) < 30:
                try: self.root.update()
                except Exception: break
                time.sleep(0.1)
        try:
            if getattr(self.review_tab, "running", False):
                self.review_tab.stop_flag = True
                self.status.set("正在停止学习循环...")
                t0 = time.time()
                while self.review_tab.running and (time.time() - t0) < 15:
                    try: self.root.update()
                    except Exception: break
                    time.sleep(0.1)
        except Exception: pass

        self._monitor_running = False
        self._backup_best_pt()
        try: self.error_book.save()
        except Exception: pass
        try: self.ground_truth.save()
        except Exception: pass
        try: self.merge_log.save()
        except Exception: pass
        try: self.guidance.save()
        except Exception: pass
        try: self.learning_rules.save()
        except Exception: pass
        try: self.root.destroy()
        except Exception: pass
        os._exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()