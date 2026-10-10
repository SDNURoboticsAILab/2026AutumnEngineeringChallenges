# review_dialogs.py
# 开卷学习的弹窗：对比窗口 / 学习报告 / 人工错误清单
# v1.5:
#   - 对比窗口：自动最大化、ESC、只看错误
#   - 人工引导（补漏 / 改类 / 合并）
#   - 学习规则（禁区 / 修正 / 漏检点 / 偏大 / 偏小 / 收敛）
#   - 自动诊断区
#   - 快捷键 1~7
import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

from core import (CLASS_NAMES, CLASS_NAME_TO_ID, compute_iou,
                  merge_boxes_union, merge_boxes_wbf)


SHIFT_MASK = 0x0001


class ReviewDialogsMixin:
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
            try: return ImageFont.truetype(name, size)
            except Exception: continue
        return ImageFont.load_default()

    def _match_pred_gt(self, preds, gt_boxes, iou_thr=0.5):
        n_gt = len(gt_boxes); n_pred = len(preds)
        if n_gt == 0 or n_pred == 0: return set(), set()
        ious = [[0.0]*n_pred for _ in range(n_gt)]
        for i, g in enumerate(gt_boxes):
            for j, p in enumerate(preds):
                ious[i][j] = compute_iou(g["xyxy"], p["xyxy"])
        pairs = []
        for i in range(n_gt):
            for j in range(n_pred):
                if ious[i][j] >= iou_thr and int(gt_boxes[i]["cls_id"]) == int(preds[j]["cls_id"]):
                    pairs.append((ious[i][j], i, j))
        pairs.sort(reverse=True)
        mp, mg = set(), set()
        for _, i, j in pairs:
            if i in mg or j in mp: continue
            mg.add(i); mp.add(j)
        return mp, mg

    # ==================== 对比窗口 ====================
    def _open_compare_window(self, key, before_m=None, after_m=None):
        gt_rec = self.app.ground_truth.get_by_key(key)
        if not gt_rec:
            messagebox.showerror("错误", f"GT 不存在: {key}"); return
        img_path = self.app.ground_truth.img_dir / gt_rec["file"]
        if not img_path.exists():
            messagebox.showerror("错误", f"图片不存在: {gt_rec['file']}"); return
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("打开失败", str(e)); return
        gt_boxes = gt_rec["boxes"]
        img_name = Path(gt_rec["file"]).name

        win = tk.Toplevel(self)
        win.title(f"对比: {key}")
        win.geometry("1300x900")
        win.resizable(True, True)
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        win.bind("<Escape>", lambda e: win.destroy())
        try:
            win.after(50, lambda: win.state('zoomed'))
        except Exception:
            pass

        # ===== 顶部 =====
        top = ttk.Frame(win); top.pack(fill=tk.X, padx=8, pady=6)
        mode_var = tk.StringVar(value="pred_vs_gt")
        has_both = (before_m is not None and after_m is not None)
        if has_both:
            ttk.Radiobutton(top, text="学习前 vs 学习后", variable=mode_var,
                            value="before_after").pack(side=tk.LEFT, padx=4)
            ttk.Radiobutton(top, text="模型预测 vs 标准答案", variable=mode_var,
                            value="pred_vs_gt").pack(side=tk.LEFT, padx=4)
        else:
            ttk.Label(top, text="模型预测 vs 标准答案",
                      font=('Consolas', 11, 'bold')).pack(side=tk.LEFT, padx=4)

        view_mode_var = tk.StringVar(value="all")
        ttk.Label(top, text="  显示:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Radiobutton(top, text="全部", variable=view_mode_var,
                        value="all").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(top, text="只看错误", variable=view_mode_var,
                        value="errors").pack(side=tk.LEFT, padx=2)

        ttk.Button(top, text="去修改GT",
                   command=lambda: self._goto_edit_gt(key, win)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="关闭", command=win.destroy).pack(side=tk.RIGHT, padx=4)

        # ===== 人工引导模式 =====
        guide_row = ttk.LabelFrame(win, text="人工引导（不改 GT，只影响训练和推理）")
        guide_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        guide_mode_var = tk.StringVar(value="off")
        for txt, val in [("关闭", "off"), ("补漏", "missed"),
                          ("改类", "wrong_cls"), ("合并", "merge")]:
            ttk.Radiobutton(guide_row, text=txt, variable=guide_mode_var,
                            value=val).pack(side=tk.LEFT, padx=6)
        guide_status = tk.StringVar(value="引导关闭")
        ttk.Label(guide_row, textvariable=guide_status,
                  font=('Consolas', 10), foreground="#cc6600").pack(side=tk.RIGHT, padx=8)

        # ===== 学习规则按钮 =====
        rule_row = ttk.LabelFrame(win, text="学习规则（快捷键 1~7 对应左图选中框）")
        rule_row.pack(fill=tk.X, padx=8, pady=(0, 4))

        rule_stat = tk.StringVar(value="")
        ttk.Label(rule_row, textvariable=rule_stat,
                  font=('Consolas', 10), foreground="#cc0066").pack(side=tk.RIGHT, padx=8)

        def _refresh_rule_stat():
            rules = getattr(self.app, "learning_rules", None)
            if rules is None:
                rule_stat.set("rules 未初始化"); return
            rule_stat.set(
                f"禁区 {rules.count_forbidden()}  |  "
                f"修正 {rules.count_corrections()}  |  "
                f"漏检 {rules.count_miss_points()}  |  "
                f"偏大/小 {rules.count_box_adjustments()}  |  "
                f"收敛 {rules.count_convergences()}")

        # ===== 主区 =====
        main = ttk.PanedWindow(win, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        left_frame = ttk.LabelFrame(main, text="模型预测"); main.add(left_frame, weight=1)
        left_canvas = tk.Canvas(left_frame, bg='#1e1e1e', highlightthickness=0)
        left_canvas.pack(fill=tk.BOTH, expand=True)
        right_frame = ttk.LabelFrame(main, text="标准答案 (GT)"); main.add(right_frame, weight=1)
        right_canvas = tk.Canvas(right_frame, bg='#1e1e1e', highlightthickness=0)
        right_canvas.pack(fill=tk.BOTH, expand=True)

        # ===== 诊断文本 =====
        diag_frame = ttk.LabelFrame(win, text="逐框诊断 + 自动分类")
        diag_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        diag_text = tk.Text(diag_frame, height=10, font=('Consolas', 10),
                            bg='#f8f8f8', wrap=tk.WORD)
        diag_text.pack(fill=tk.X, padx=5, pady=5)

        # ===== 底部 =====
        bottom = ttk.Frame(win); bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        status_var = tk.StringVar(value="提示: 点击图上任意框可选中；快捷键 1~7")
        ttk.Label(bottom, textvariable=status_var,
                  font=('Consolas', 10), foreground="#007acc").pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="关闭", command=win.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="🎯 高强度训练",
                   command=lambda: (win.destroy(), self.intensive_train())
                   ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="❌ 标记选中框为多余",
                   command=lambda: _mark_extra()).pack(side=tk.RIGHT, padx=4)

        # ===== 选中状态 =====
        selected = {"panel": None, "idx": None, "indices_left": [],
                    "drag_start": None, "pending_mode": None}
        guide_boxes = {"missed": [], "wrong_cls": [], "merge": []}

        # ===== 学习规则动作 =====
        def _require_left_box():
            if selected["panel"] != "left":
                messagebox.showinfo("提示", "请先点击左图的一个预测框")
                return None
            preds = (after_m or before_m or {}).get("_preds", [])
            if not (0 <= selected["idx"] < len(preds)):
                return None
            return preds[selected["idx"]]

        def _mark_forbidden():
            rules = getattr(self.app, "learning_rules", None)
            if rules is None:
                messagebox.showerror("错误", "learning_rules 未初始化"); return
            p = _require_left_box()
            if p is None: return
            added, skipped = rules.add_forbidden_from_box(
                img_name, p["xyxy"], gt_boxes=gt_boxes,
                img_size=img.size, source="manual")
            _refresh_rule_stat(); redraw()
            if added == 0:
                messagebox.showwarning("无法设为禁区",
                    "4 个顶点都太靠近标准答案，无法生成安全禁区。")
            else:
                msg = f"已生成 {added} 个禁区圆"
                if skipped: msg += f"（{skipped} 个顶点被跳过）"
                status_var.set(msg)
            selected["panel"] = None; selected["idx"] = None

        def _start_correction():
            rules = getattr(self.app, "learning_rules", None)
            if rules is None:
                messagebox.showerror("错误", "learning_rules 未初始化"); return
            p = _require_left_box()
            if p is None: return
            _open_correction_dialog(p)

        def _open_correction_dialog(old_pred):
            dlg = tk.Toplevel(win)
            dlg.title("修正标签")
            dlg.geometry("460x280")
            dlg.resizable(True, True)
            dlg.bind("<Escape>", lambda e: dlg.destroy())

            ttk.Label(dlg, text="用修正后的框替换这个预测框：",
                      font=('Consolas', 10, 'bold')).pack(pady=8)

            info = ttk.Frame(dlg); info.pack(fill=tk.X, padx=12, pady=4)
            ttk.Label(info, text=f"旧框: {[int(v) for v in old_pred['xyxy']]}",
                      font=('Consolas', 9)).pack(anchor=tk.W)
            ttk.Label(info, text=f"旧类别: {old_pred['cls_name']}",
                      font=('Consolas', 9)).pack(anchor=tk.W)

            row = ttk.Frame(dlg); row.pack(fill=tk.X, padx=12, pady=8)
            ttk.Label(row, text="新框 x1 y1 x2 y2:").pack(side=tk.LEFT)
            x1v = tk.IntVar(value=int(old_pred["xyxy"][0]))
            y1v = tk.IntVar(value=int(old_pred["xyxy"][1]))
            x2v = tk.IntVar(value=int(old_pred["xyxy"][2]))
            y2v = tk.IntVar(value=int(old_pred["xyxy"][3]))
            for v in (x1v, y1v, x2v, y2v):
                ttk.Entry(row, textvariable=v, width=6).pack(side=tk.LEFT, padx=2)

            row2 = ttk.Frame(dlg); row2.pack(fill=tk.X, padx=12, pady=8)
            ttk.Label(row2, text="新类别:").pack(side=tk.LEFT)
            cls_v = tk.StringVar(value=old_pred["cls_name"])
            ttk.Combobox(row2, textvariable=cls_v, width=12, state="readonly",
                         values=["obstacle", "cola", "football"]).pack(side=tk.LEFT, padx=4)

            def ok():
                new_xyxy = [x1v.get(), y1v.get(), x2v.get(), y2v.get()]
                cid = CLASS_NAME_TO_ID.get(cls_v.get(), 0)
                rules.add_label_correction(
                    img_name, old_pred["xyxy"], new_xyxy,
                    cid, cls_v.get(), source="manual")
                _refresh_rule_stat(); redraw(); dlg.destroy()
                status_var.set(f"✅ 已修正: {old_pred['cls_name']} → {cls_v.get()}")

            btns = ttk.Frame(dlg); btns.pack(pady=8)
            ttk.Button(btns, text="确定", command=ok).pack(side=tk.LEFT, padx=4)
            ttk.Button(btns, text="取消", command=dlg.destroy).pack(side=tk.LEFT, padx=4)

        def _adjust(kind):
            """偏大/偏小：把预测框替换成匹配的 GT 框"""
            rules = getattr(self.app, "learning_rules", None)
            if rules is None:
                messagebox.showerror("错误", "learning_rules 未初始化"); return
            p = _require_left_box()
            if p is None: return
            # 找匹配的 GT（同类、IoU 最高）
            best_g = None; best_iou = 0
            for g in gt_boxes:
                if int(g["cls_id"]) != int(p["cls_id"]): continue
                iou = compute_iou(p["xyxy"], g["xyxy"])
                if iou > best_iou:
                    best_iou = iou; best_g = g
            if best_g is None:
                messagebox.showwarning("没找到匹配 GT",
                    "找不到同类的 GT 框，无法判断目标位置。")
                return
            rules.add_box_adjustment(
                img_name, kind, p["xyxy"], best_g["xyxy"], source="manual")
            _refresh_rule_stat(); redraw()
            status_var.set(f"已标记{ '偏大' if kind=='too_big' else '偏小' }，"
                           f"新框来自 GT")

        def _start_convergence():
            """收敛：先点选误检，再拖拽画目标位置"""
            rules = getattr(self.app, "learning_rules", None)
            if rules is None:
                messagebox.showerror("错误", "learning_rules 未初始化"); return
            p = _require_left_box()
            if p is None: return
            selected["pending_mode"] = "convergence"
            selected["pending_from"] = p
            guide_mode_var.set("missed")   # 复用拖拽画框
            status_var.set("🔄 收敛模式：在左图拖拽画出正确位置")

        def _do_convergence_to(target_xyxy):
            p = selected.get("pending_from")
            if p is None: return
            rules = getattr(self.app, "learning_rules", None)
            if rules is None: return
            # 找匹配的 GT（同类、IoU 最高）
            best_g = None; best_iou = 0
            for g in gt_boxes:
                if int(g["cls_id"]) != int(p["cls_id"]): continue
                iou = compute_iou(target_xyxy, g["xyxy"])
                if iou > best_iou:
                    best_iou = iou; best_g = g
            to_cls = best_g["cls_name"] if best_g else p["cls_name"]
            rules.add_convergence(
                img_name, p["xyxy"], target_xyxy,
                p["cls_name"], to_cls, source="manual")
            _refresh_rule_stat(); redraw()
            status_var.set(f"🔄 已记录收敛: {p['cls_name']} → {to_cls}")
            selected["pending_mode"] = None
            selected["pending_from"] = None
            selected["panel"] = None; selected["idx"] = None

        # ===== 绑定按钮 =====
        ttk.Button(rule_row, text="🚫 禁区(4)",
                   command=_mark_forbidden).pack(side=tk.LEFT, padx=2)
        ttk.Button(rule_row, text="◀ 偏小(2)",
                   command=lambda: _adjust("too_small")).pack(side=tk.LEFT, padx=2)
        ttk.Button(rule_row, text="▶ 偏大(3)",
                   command=lambda: _adjust("too_big")).pack(side=tk.LEFT, padx=2)
        ttk.Button(rule_row, text="✗ 多余(5)",
                   command=_mark_forbidden).pack(side=tk.LEFT, padx=2)
        ttk.Button(rule_row, text="＋ 缺失(6)",
                   command=lambda: guide_mode_var.set("missed")
                   ).pack(side=tk.LEFT, padx=2)
        ttk.Button(rule_row, text="🔄 收敛(7)",
                   command=_start_convergence).pack(side=tk.LEFT, padx=2)
        ttk.Button(rule_row, text="✏️ 修正",
                   command=_start_correction).pack(side=tk.LEFT, padx=2)
        ttk.Button(rule_row, text="📋 查看规则",
                   command=lambda: _show_rules_dialog()).pack(side=tk.LEFT, padx=2)

        def _show_rules_dialog():
            rules = getattr(self.app, "learning_rules", None)
            if rules is None: return
            dlg = tk.Toplevel(win)
            dlg.title(f"学习规则 - {img_name}")
            dlg.geometry("720x600")
            dlg.resizable(True, True)
            dlg.bind("<Escape>", lambda e: dlg.destroy())

            ttk.Label(dlg, text=f"图片: {img_name}",
                      font=('Consolas', 11, 'bold')).pack(anchor=tk.W, padx=10, pady=6)

            sections = [
                ("🚫 禁区圆", rules.get_forbidden_zones(img_name),
                 lambda z: f"  ({z['cx']:.0f},{z['cy']:.0f}) r={z['r']:.1f} [{z.get('source','')}]"),
                ("✏️ 标签修正", rules.get_corrections(img_name),
                 lambda c: f"  {c['old_xyxy']} → {c['new_xyxy']} 类别: {c['new_cls_name']} [{c.get('source','')}]"),
                ("📍 漏检点", rules.get_miss_points(img_name),
                 lambda m: f"  {m['xyxy']} 类别: {m['cls_name']} repeat={m['repeat']} [{m.get('source','')}]"),
                ("▶◀ 偏大/偏小", rules.get_box_adjustments(img_name),
                 lambda a: f"  {a['kind']}: {a['old_xyxy']} → {a['new_xyxy']} [{a.get('source','')}]"),
                ("🔄 收敛", rules.get_convergences(img_name),
                 lambda v: f"  {v['from_xyxy']} → {v['to_xyxy']}  {v['from_cls']}→{v['to_cls']} [{v.get('source','')}]"),
            ]
            for title, items, fmt in sections:
                ttk.Label(dlg, text=f"{title}: {len(items)} 个",
                          font=('Consolas', 10, 'bold'),
                          foreground="#cc0066").pack(anchor=tk.W, padx=10, pady=(6, 0))
                t = tk.Text(dlg, height=4, font=('Consolas', 9))
                t.pack(fill=tk.X, padx=10, pady=2)
                for i in items:
                    t.insert(tk.END, fmt(i) + "\n")
                t.config(state=tk.DISABLED)

            def clear_this():
                if not messagebox.askyesno("确认", f"清空 {img_name} 所有规则？"):
                    return
                rules.clear_forbidden(img_name)
                rules.clear_corrections(img_name)
                rules.clear_miss_points(img_name)
                rules.clear_box_adjustments(img_name)
                rules.clear_convergences(img_name)
                _refresh_rule_stat(); redraw(); dlg.destroy()

            ttk.Button(dlg, text="清空本图规则",
                       command=clear_this).pack(pady=8)

        _refresh_rule_stat()

        # ===== 状态刷新 =====
        def _refresh_status():
            gm = guide_mode_var.get()
            if gm == "off":
                guide_status.set("引导关闭")
            elif gm == "missed":
                if selected.get("pending_mode") == "convergence":
                    guide_status.set("收敛模式：拖拽画目标位置")
                else:
                    guide_status.set("补漏：在空白处拖拽画框")
            elif gm == "wrong_cls":
                guide_status.set("改类：点击预测框，然后选择新类别")
            elif gm == "merge":
                guide_status.set("合并：Shift 多选预测框，点右键合并")

            if selected["panel"] is None:
                status_var.set("提示: 点击图上任意框可选中；快捷键 1~7")
            elif selected["panel"] == "left":
                preds = (after_m or before_m or {}).get("_preds", [])
                if 0 <= selected["idx"] < len(preds):
                    p = preds[selected["idx"]]
                    status_var.set(
                        f"已选: 左图预测框 #{selected['idx']} "
                        f"类别={p['cls_name']} conf={p.get('conf',1.0):.2f}  |  "
                        f"1正确 2偏小 3偏大 4禁区 5多余 6缺失 7收敛")
            else:
                if 0 <= selected["idx"] < len(gt_boxes):
                    g = gt_boxes[selected["idx"]]
                    status_var.set(f"已选: 右图 GT 框 #{selected['idx']} 类别={g['cls_name']}")

        # ===== 命中测试 / 坐标 =====
        def _hit_test(canvas, event, boxes):
            iw, ih = img.size
            cw = canvas.winfo_width(); ch = canvas.winfo_height()
            if cw < 50 or ch < 50: return None
            scale = min(cw / iw, ch / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            ox = (cw - nw) // 2; oy = (ch - nh) // 2
            ix = (event.x - ox) / scale
            iy = (event.y - oy) / scale
            for j in range(len(boxes) - 1, -1, -1):
                x1, y1, x2, y2 = boxes[j]["xyxy"]
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    return j
            return None

        def _canvas_to_img(canvas, event):
            iw, ih = img.size
            cw = canvas.winfo_width(); ch = canvas.winfo_height()
            if cw < 50 or ch < 50: return None, None
            scale = min(cw / iw, ch / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            ox = (cw - nw) // 2; oy = (ch - nh) // 2
            return (event.x - ox) / scale, (event.y - oy) / scale

        # ===== 事件 =====
        def _on_left_click(event):
            gm = guide_mode_var.get()
            if gm == "missed":
                selected["drag_start"] = _canvas_to_img(left_canvas, event)
                return
            preds = (after_m or before_m or {}).get("_preds", [])
            hit = _hit_test(left_canvas, event, preds)
            if gm == "wrong_cls" and hit is not None:
                _do_wrong_cls(hit); return
            if gm == "merge" and hit is not None:
                is_shift = bool(event.state & SHIFT_MASK)
                if is_shift:
                    if hit in selected["indices_left"]:
                        selected["indices_left"].remove(hit)
                    else:
                        selected["indices_left"].append(hit)
                else:
                    selected["indices_left"] = [hit]
                selected["panel"] = "left"; selected["idx"] = hit
                _refresh_status(); redraw(); return
            selected["panel"] = "left" if hit is not None else None
            selected["idx"] = hit
            if hit is None: selected["indices_left"] = []
            _refresh_status(); redraw()

        def _on_right_click(event):
            gm = guide_mode_var.get()
            if gm == "merge" and len(selected["indices_left"]) >= 2:
                _do_merge(); return
            hit = _hit_test(right_canvas, event, gt_boxes)
            selected["panel"] = "right" if hit is not None else None
            selected["idx"] = hit
            _refresh_status(); redraw()

        def _on_drag(event):
            if guide_mode_var.get() != "missed": return
            if selected.get("drag_start") is None: return
            x1, y1 = selected["drag_start"]
            x2, y2 = _canvas_to_img(left_canvas, event)
            if x2 is None: return
            iw, ih = img.size
            cw = left_canvas.winfo_width(); ch = left_canvas.winfo_height()
            scale = min(cw / iw, ch / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            ox = (cw - nw) // 2; oy = (ch - nh) // 2
            left_canvas.delete("drag_rect")
            left_canvas.create_rectangle(
                ox + x1 * scale, oy + y1 * scale,
                ox + x2 * scale, oy + y2 * scale,
                outline='#ff00ff', width=3, tags="drag_rect")

        def _on_release(event):
            if guide_mode_var.get() != "missed": return
            if selected.get("drag_start") is None: return
            x1, y1 = selected["drag_start"]
            x2, y2 = _canvas_to_img(left_canvas, event)
            selected["drag_start"] = None
            left_canvas.delete("drag_rect")
            if x2 is None or abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                return
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            xyxy = [x1, y1, x2, y2]
            if selected.get("pending_mode") == "convergence":
                _do_convergence_to(xyxy)
            else:
                _do_missed(xyxy)

        def _do_missed(xyxy):
            cls_win = tk.Toplevel(win)
            cls_win.title("补漏 - 选类别")
            cls_win.geometry("300x160")
            cls_win.bind("<Escape>", lambda e: cls_win.destroy())
            ttk.Label(cls_win, text="选择这个漏检框的类别:").pack(pady=8)
            v = tk.StringVar(value="obstacle")
            for c in ("obstacle", "cola", "football"):
                ttk.Radiobutton(cls_win, text=c, variable=v, value=c
                                ).pack(anchor=tk.W, padx=20)
            def ok():
                cid = CLASS_NAME_TO_ID.get(v.get(), 0)
                guide_boxes["missed"].append({
                    "xyxy": [float(x) for x in xyxy],
                    "cls_id": cid, "cls_name": v.get()})
                try:
                    self.app.guidance.add_missed(img_name, xyxy, cid, v.get())
                except Exception as e:
                    print(f"[guidance 保存失败] {e}")
                rules = getattr(self.app, "learning_rules", None)
                if rules is not None:
                    try:
                        rules.add_miss_point(img_name, xyxy, cid, v.get(),
                                             source="manual")
                    except Exception as e:
                        print(f"[rules 保存失败] {e}")
                _refresh_rule_stat(); cls_win.destroy()
                _refresh_status(); redraw()
            ttk.Button(cls_win, text="确定", command=ok).pack(pady=6)

        def _do_wrong_cls(pred_idx):
            preds = (after_m or before_m or {}).get("_preds", [])
            if not (0 <= pred_idx < len(preds)): return
            p = preds[pred_idx]
            cls_win = tk.Toplevel(win)
            cls_win.title("改类")
            cls_win.geometry("300x180")
            cls_win.bind("<Escape>", lambda e: cls_win.destroy())
            ttk.Label(cls_win, text=f"当前: {p['cls_name']}\n改为:").pack(pady=8)
            v = tk.StringVar(value=p["cls_name"])
            for c in ("obstacle", "cola", "football"):
                ttk.Radiobutton(cls_win, text=c, variable=v, value=c
                                ).pack(anchor=tk.W, padx=20)
            def ok():
                if v.get() == p["cls_name"]:
                    cls_win.destroy(); return
                guide_boxes["wrong_cls"].append({
                    "pred_xyxy": list(p["xyxy"]),
                    "from_cls": p["cls_name"], "to_cls": v.get()})
                try:
                    self.app.guidance.add_wrong_cls(
                        img_name, p["xyxy"], p["cls_name"], v.get())
                except Exception as e:
                    print(f"[guidance 保存失败] {e}")
                cls_win.destroy(); _refresh_status(); redraw()
            ttk.Button(cls_win, text="确定", command=ok).pack(pady=6)

        def _do_merge():
            preds = (after_m or before_m or {}).get("_preds", [])
            idxs = selected["indices_left"]
            boxes = [preds[i] for i in idxs if 0 <= i < len(preds)]
            if len(boxes) < 2: return
            answer = messagebox.askyesnocancel(
                "合并方式",
                "「是」= 加权融合(WBF)\n「否」= 并集外接矩形\n「取消」= 放弃")
            if answer is None: return
            merged = merge_boxes_wbf(boxes) if answer else merge_boxes_union(boxes)
            if merged is None: return
            group = [list(b["xyxy"]) for b in boxes]
            guide_boxes["merge"].append({
                "group": group,
                "merged_xyxy": list(merged["xyxy"]),
                "cls_id": merged["cls_id"],
                "cls_name": merged["cls_name"]})
            try:
                self.app.guidance.add_duplicate(
                    img_name, group, merged["xyxy"],
                    merged["cls_id"], merged["cls_name"])
            except Exception as e:
                print(f"[guidance 保存失败] {e}")
            selected["indices_left"] = []
            _refresh_status(); redraw()

        left_canvas.bind("<Button-1>", _on_left_click)
        left_canvas.bind("<B1-Motion>", _on_drag)
        left_canvas.bind("<ButtonRelease-1>", _on_release)
        left_canvas.bind("<Button-3>", _on_right_click)
        right_canvas.bind("<Button-1>", _on_right_click)

        # ===== 快捷键 1~7 =====
        def _on_key(event):
            k = event.keysym
            if k == "1":
                # 正确：清空该预测框的所有规则（保留原状）
                status_var.set("✅ 标记正确（无需处理）")
                selected["panel"] = None; selected["idx"] = None
                redraw()
            elif k == "2":
                _adjust("too_small")
            elif k == "3":
                _adjust("too_big")
            elif k == "4":
                _mark_forbidden()
            elif k == "5":
                _mark_forbidden()
            elif k == "6":
                guide_mode_var.set("missed")
            elif k == "7":
                _start_convergence()
        win.bind("<Key>", _on_key)
        win.focus_set()

        # ===== 标记多余 =====
        def _mark_extra():
            if selected["panel"] is None:
                messagebox.showinfo("提示", "请先在图上点击一个框选中它")
                return
            if selected["panel"] == "left":
                preds = (after_m or before_m or {}).get("_preds", [])
                if not (0 <= selected["idx"] < len(preds)): return
                p = preds[selected["idx"]]
                if not messagebox.askyesno("确认",
                    f"把左侧预测框标记为「多余」?\n\n类别: {p['cls_name']}"):
                    return
                self._add_manual_error(key, p["cls_name"], p.get("conf", 1.0), "extra")
                self.app.ground_truth.update_score(key, 0.0, "fail")
                self._refresh_all_gt_views()
                status_var.set(f"✅ 已标记「{p['cls_name']}」为多余")
                selected["panel"] = None; selected["idx"] = None
                redraw()
            else:
                gi = selected["idx"]
                if not (0 <= gi < len(gt_boxes)): return
                g = gt_boxes[gi]
                if not messagebox.askyesno("确认",
                    f"从 GT 中删除这个框？\n\n类别: {g['cls_name']}"):
                    return
                new_boxes = [b for i, b in enumerate(gt_boxes) if i != gi]
                boxes_for_save = [{
                    "xyxy": b["xyxy"], "cls_id": b["cls_id"],
                    "cls_name": b["cls_name"],
                    "required": b.get("required", False),
                } for b in new_boxes]
                self.app.ground_truth.update_boxes(key, boxes_for_save, reviewed=True)
                self._refresh_all_gt_views()
                messagebox.showinfo("已修改", "GT 已更新，请关闭本窗口重新查看")
                win.destroy()

        left_photo_holder = [None]
        right_photo_holder = [None]

        # ===== 绘制 =====
        def draw_panel(canvas, preds, photo_holder, panel_name, view_mode):
            img2 = img.copy(); draw = ImageDraw.Draw(img2)
            iw, ih = img2.size; fs = max(12, min(18, iw // 70))
            font = self._get_font(fs)
            mp, mg = self._match_pred_gt(preds, gt_boxes) if (preds and gt_boxes) else (set(), set())
            show_all = (view_mode == "all")

            # GT 框
            for i, g in enumerate(gt_boxes):
                is_matched = (i in mg)
                if not show_all and is_matched: continue
                x1, y1, x2, y2 = g["xyxy"]
                is_req = g.get("required", False)
                is_sel = (panel_name == "right" and selected["panel"] == "right"
                          and selected["idx"] == i)
                if is_sel: color, width = '#FFFF00', 4
                elif is_matched: color, width = '#00cc00', 3
                else: color, width = '#ff3333', 3
                label = f"GT:{g['cls_name']}" + (" ★" if is_req else "")
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                try: bbox = draw.textbbox((0, 0), label, font=font)
                except Exception: bbox = (0, 0, len(label)*fs, fs)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                ty = max(0, y1 - th - 4)
                draw.rectangle([x1, ty, x1+tw+4, ty+th+4], fill=color)
                draw.text((x1+2, ty+2), label, fill='white', font=font)

            # 预测框
            for j, p in enumerate(preds or []):
                is_matched = (j in mp)
                if not show_all and is_matched: continue
                x1, y1, x2, y2 = p["xyxy"]
                is_sel = (panel_name == "left" and (
                    selected["idx"] == j or j in selected.get("indices_left", [])))
                if is_sel: color, width = '#FFFF00', 4
                elif is_matched: color, width = '#0099ff', 2
                else: color, width = '#ff66cc', 2
                label = f"P:{p['cls_name']} {p.get('conf',1.0):.2f}"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                try: bbox = draw.textbbox((0, 0), label, font=font)
                except Exception: bbox = (0, 0, len(label)*fs, fs)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                ty = min(ih - th - 4, y2 + 2)
                draw.rectangle([x1, ty, x1+tw+4, ty+th+4], fill=color)
                draw.text((x1+2, ty+2), label, fill='white', font=font)

            # 引导补漏（紫色虚线）
            for ms in guide_boxes["missed"]:
                x1, y1, x2, y2 = ms["xyxy"]
                for seg in [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]:
                    draw.line(seg, fill='#cc00ff', width=3)

            # 禁区圆（红色）
            rules = getattr(self.app, "learning_rules", None)
            if rules is not None and panel_name == "left":
                for z in rules.get_forbidden_zones(img_name):
                    cx, cy, r = z["cx"], z["cy"], z["r"]
                    pts = []
                    for k in range(24):
                        a = k * math.pi / 12
                        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                    for k in range(len(pts)):
                        a = pts[k]; b = pts[(k+1) % len(pts)]
                        draw.line([a, b], fill='#ff0000', width=2)

            # 收敛箭头
            if rules is not None and panel_name == "left":
                for v in rules.get_convergences(img_name):
                    from_xyxy = v["from_xyxy"]; to_xyxy = v["to_xyxy"]
                    # from 灰色虚线
                    for seg in [(from_xyxy[0], from_xyxy[1], from_xyxy[2], from_xyxy[1]),
                                (from_xyxy[2], from_xyxy[1], from_xyxy[2], from_xyxy[3]),
                                (from_xyxy[2], from_xyxy[3], from_xyxy[0], from_xyxy[3]),
                                (from_xyxy[0], from_xyxy[3], from_xyxy[0], from_xyxy[1])]:
                        draw.line(seg, fill='#888888', width=2)
                    # to 紫色实线
                    draw.rectangle(to_xyxy, outline='#cc00ff', width=3)
                    # 箭头
                    fcx = (from_xyxy[0] + from_xyxy[2]) / 2
                    fcy = (from_xyxy[1] + from_xyxy[3]) / 2
                    tcx = (to_xyxy[0] + to_xyxy[2]) / 2
                    tcy = (to_xyxy[1] + to_xyxy[3]) / 2
                    draw.line([(fcx, fcy), (tcx, tcy)], fill='#cc00ff', width=2)

            # 右上角统计
            n_gt = len(gt_boxes); n_pred = len(preds) if preds else 0
            stat = f"GT {n_gt}  预测 {n_pred}  命中 {len(mg)}  漏检 {n_gt-len(mg)}  误检 {n_pred-len(mp)}"
            try: bbox = draw.textbbox((0, 0), stat, font=font)
            except Exception: bbox = (0, 0, 400, 24)
            sw_ = bbox[2] - bbox[0] + 16
            draw.rectangle([iw - sw_ - 8, 8, iw - 8, 38], fill='#222222')
            draw.text((iw - sw_ - 4, 14), stat, fill='white', font=font)

            canvas.update_idletasks()
            cw = canvas.winfo_width(); ch = canvas.winfo_height()
            if cw < 50 or ch < 50: cw, ch = 550, 600
            scale = min(cw / iw, ch / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            resized = img2.resize((nw, nh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            canvas.delete("all")
            canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor=tk.NW, image=photo)
            photo_holder[0] = photo

        def render_diag_to(text_widget, m, prefix=""):
            if m is None:
                text_widget.insert(tk.END, f"{prefix}无数据\n"); return
            correct = m.get("correct", 0); wrong_cls = m.get("wrong_cls", 0)
            missed_n = m.get("missed_n", 0); extra_n = m.get("extra_n", 0)
            dup_n = m.get("dup_n", 0)
            total = correct + wrong_cls + missed_n
            f1_val = m.get("f1", 0)
            violated = m.get("required_violated", False)
            suffix = "  [必要框漏检，0 分]" if violated else ""
            text_widget.insert(tk.END,
                f"{prefix}正确 {correct}/{total}  漏检 {missed_n}  "
                f"误检 {extra_n}  重复 {dup_n}  类别错 {wrong_cls}  F1={f1_val:.3f}{suffix}\n")

            ae = m.get("_auto_errors")
            if ae:
                gt_boxes_local = m.get("_gt_boxes", gt_boxes)
                preds_local = m.get("_preds", [])
                # 偏大/偏小
                for gi, pj, ratio in ae.get("too_big", []):
                    text_widget.insert(tk.END,
                        f"  ⚠ 偏大: GT#{gi} {gt_boxes_local[gi]['cls_name']} "
                        f"← P#{pj}（{ratio:.2f}x）\n")
                for gi, pj, ratio in ae.get("too_small", []):
                    text_widget.insert(tk.END,
                        f"  ⚠ 偏小: GT#{gi} {gt_boxes_local[gi]['cls_name']} "
                        f"← P#{pj}（{ratio:.2f}x）\n")
                # 多余
                for pj in ae.get("extra", []):
                    if pj < len(preds_local):
                        p = preds_local[pj]
                        text_widget.insert(tk.END,
                            f"  ✗ 多余: P#{pj} {p['cls_name']} "
                            f"({p.get('conf',1.0):.2f})\n")
                # 缺失
                for gi in ae.get("missed", []):
                    if gi < len(gt_boxes_local):
                        g = gt_boxes_local[gi]
                        text_widget.insert(tk.END,
                            f"  ✗ 缺失: GT#{gi} {g['cls_name']}\n")
                # 类别错
                for gi, pj, iou, g_cls, p_cls in ae.get("cls_error", []):
                    text_widget.insert(tk.END,
                        f"  ✗ 类别错: GT#{gi} {g_cls} → P#{pj} {p_cls}  (IoU={iou:.2f})\n")
                # 重复
                for pj in ae.get("duplicate", []):
                    text_widget.insert(tk.END, f"  ⚠ 重复: P#{pj}\n")
                # 收敛
                if ae.get("trigger_convergence"):
                    text_widget.insert(tk.END,
                        f"  🔄 触发自动收敛（剩余==误检 且 误检==漏检）\n")
                for ei, mi in ae.get("convergence", []):
                    if ei < len(preds_local) and mi < len(gt_boxes_local):
                        p = preds_local[ei]; g = gt_boxes_local[mi]
                        text_widget.insert(tk.END,
                            f"  🔄 收敛: P#{ei} {p['cls_name']} → GT#{mi} {g['cls_name']}\n")

        def redraw():
            if not win.winfo_exists(): return
            mode = mode_var.get()
            view_mode = view_mode_var.get()
            diag_text.config(state="normal")
            diag_text.delete("1.0", tk.END)
            if mode == "before_after" and has_both:
                left_frame.config(text="学习前预测")
                right_frame.config(text="学习后预测")
                draw_panel(left_canvas, before_m.get("_preds", []),
                           left_photo_holder, "left", view_mode)
                draw_panel(right_canvas, after_m.get("_preds", []),
                           right_photo_holder, "right", view_mode)
                b_f1 = before_m.get("f1", 0); a_f1 = after_m.get("f1", 0)
                diag_text.insert(tk.END, f"学习前 F1: {b_f1:.3f}\n")
                diag_text.insert(tk.END, f"学习后 F1: {a_f1:.3f}\n")
                diag_text.insert(tk.END, f"变化: {a_f1-b_f1:+.3f}\n\n")
                render_diag_to(diag_text, before_m, prefix="[学习前] ")
                diag_text.insert(tk.END, "\n")
                render_diag_to(diag_text, after_m, prefix="[学习后] ")
            else:
                preds_m = after_m or before_m
                left_frame.config(text="模型预测")
                right_frame.config(text="标准答案 (GT)")
                preds = preds_m.get("_preds", []) if preds_m else []
                draw_panel(left_canvas, preds, left_photo_holder, "left", view_mode)
                draw_panel(right_canvas, [], right_photo_holder, "right", view_mode)
                if preds_m: render_diag_to(diag_text, preds_m, prefix="")
                else: diag_text.insert(tk.END, "暂无模型预测数据\n")
            diag_text.config(state="disabled")

        mode_var.trace_add('write', lambda *a: redraw())
        view_mode_var.trace_add('write', lambda *a: redraw())
        guide_mode_var.trace_add('write', lambda *a: _refresh_status())
        left_canvas.bind('<Configure>', lambda e: redraw())
        right_canvas.bind('<Configure>', lambda e: redraw())
        win.after(50, redraw)

    def _goto_edit_gt(self, key, win=None):
        if win is not None:
            try: win.destroy()
            except Exception: pass
        rec = self.app.ground_truth.get_by_key(key)
        if not rec: return
        img_path = self.app.ground_truth.img_dir / rec["file"]
        if not img_path.exists():
            messagebox.showerror("错误", f"GT 图片不存在: {rec['file']}"); return
        self.app.notebook.select(self.app.gt_tab)
        self.app.gt_tab._load_gt_for_review(key, img_path, rec["boxes"])
        self.app.status.set(f"请在 GT 管理页修改: {key}")

    # ==================== 学习报告 ====================
    def open_learning_report(self):
        if not self._last_report:
            messagebox.showinfo("提示", "还没有学习记录。先运行一次「只考试一次」或「开卷学习闭环」")
            return
        report = self._last_report
        b = report["before"]; a = report["after"]
        if getattr(self, "_report_win", None) is not None and self._report_win.winfo_exists():
            self._report_win.destroy()
        win = tk.Toplevel(self); self._report_win = win
        win.title(f"学习报告 - {report.get('time', '')}")
        win.geometry("1100x750")
        win.resizable(True, True)
        win.bind("<Escape>", lambda e: win.destroy())
        try: win.after(50, lambda: win.state('zoomed'))
        except Exception: pass
        win.protocol("WM_DELETE_WINDOW", win.destroy)

        stat = ttk.LabelFrame(win, text="学习统计"); stat.pack(fill=tk.X, padx=8, pady=6)
        pass_b = sum(1 for v in b.values() if v.get("f1", 0) >= 0.9 - 1e-6)
        pass_a = sum(1 for v in a.values() if v.get("f1", 0) >= 0.9 - 1e-6)
        info = (f"学习前通过: {pass_b}/{len(b)}    学习后通过: {pass_a}/{len(a)}\n"
                f"学会了: {report.get('learned',0)}  "
                f"没学会: {report.get('still_fail',0)}  "
                f"遗忘: {report.get('forgot',0)}  "
                f"保持: {report.get('kept',0)}")
        ttk.Label(stat, text=info, font=('Consolas', 11)).pack(anchor=tk.W, padx=8, pady=6)

        listf = ttk.LabelFrame(win, text="逐张结果（双击查看对比）")
        listf.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        cols = ("key", "before", "after", "delta", "status")
        tree = ttk.Treeview(listf, columns=cols, show="headings", height=20)
        for c, t, w in [("key","图片",360),("before","学习前",100),
                        ("after","学习后",100),("delta","变化",90),("status","状态",120)]:
            tree.heading(c, text=t)
            tree.column(c, width=w, anchor="center" if c != "key" else "w")
        sb = ttk.Scrollbar(listf, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        tree.tag_configure("learned", foreground="#008800")
        tree.tag_configure("still", foreground="#cc0000")
        tree.tag_configure("forgot", foreground="#cc6600")
        tree.tag_configure("kept", foreground="#0066cc")

        rows = list(b.keys())
        def fill_rows(filter_mode="all"):
            for iid in tree.get_children(): tree.delete(iid)
            for key in rows:
                bf = b[key].get("f1", 0); af = a.get(key, {}).get("f1", 0)
                if filter_mode == "fail" and af >= 0.9 - 1e-6: continue
                delta = af - bf
                b_ok = bf >= 0.9 - 1e-6; a_ok = af >= 0.9 - 1e-6
                if a_ok and not b_ok: st, tag = "学会了", "learned"
                elif not a_ok and not b_ok: st, tag = "没学会", "still"
                elif not a_ok and b_ok: st, tag = "遗忘", "forgot"
                else: st, tag = "保持", "kept"
                tree.insert("", "end", iid=key,
                            values=(key, f"{bf:.3f}", f"{af:.3f}", f"{delta:+.3f}", st),
                            tags=(tag,))

        def on_dbl(event):
            sel = tree.selection()
            if not sel: return
            k = sel[0]
            self._open_compare_window(k, b.get(k), a.get(k))
        tree.bind("<Double-Button-1>", on_dbl)
        fill_rows("all")

        btns = ttk.Frame(win); btns.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(btns, text="双击行查看对比", foreground="#666").pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="只看未通过", command=lambda: fill_rows("fail")).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="显示全部", command=lambda: fill_rows("all")).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="关闭", command=win.destroy).pack(side=tk.RIGHT, padx=4)

    # ==================== 人工错误清单 ====================
    def show_manual_errors(self):
        if not self._manual_errors:
            messagebox.showinfo("提示", "还没有人工标记的错误\n\n"
                                        "在对比窗口里点选预测框 → 「❌ 标记选中框为多余」")
            return

        win = tk.Toplevel(self)
        win.title("人工错误清单")
        win.geometry("1000x600")
        win.resizable(True, True)
        win.bind("<Escape>", lambda e: win.destroy())
        win.protocol("WM_DELETE_WINDOW", win.destroy)

        info = ttk.LabelFrame(win, text="说明")
        info.pack(fill=tk.X, padx=10, pady=(10, 6))
        ttk.Label(info,
                  text="① 对比窗口点选预测框 → 「❌ 标记选中框为多余」\n"
                       "② 点「🎯 高强度训练」反复训练\n"
                       "退出条件：F1≥0.9 且连续 3 次不复发",
                  justify=tk.LEFT, font=('Consolas', 10)).pack(anchor=tk.W, padx=8, pady=6)

        stat = ttk.Frame(win); stat.pack(fill=tk.X, padx=10, pady=4)
        pending = sum(1 for v in self._manual_errors.values() if not v.get("cleared"))
        cleared = sum(1 for v in self._manual_errors.values() if v.get("cleared"))
        ttk.Label(stat, text=f"待解决: {pending}  |  已解决: {cleared}",
                  font=('Consolas', 11, 'bold'),
                  foreground="#cc6600").pack(side=tk.LEFT)

        listf = ttk.LabelFrame(win, text="清单")
        listf.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        cols = ("key", "cls", "conf", "streak", "status", "time")
        tree = ttk.Treeview(listf, columns=cols, show="headings",
                            selectmode="extended", height=14)
        for c, t, w in [("key", "图片", 280), ("cls", "错误类别", 90),
                        ("conf", "置信度", 70), ("streak", "连续OK", 70),
                        ("status", "状态", 100), ("time", "标记时间", 160)]:
            tree.heading(c, text=t)
            tree.column(c, width=w, anchor="center" if c != "key" else "w")
        tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        tree.tag_configure("cleared", foreground="#008800")
        tree.tag_configure("pending", foreground="#cc0000")

        for k, info2 in self._manual_errors.items():
            is_cleared = info2.get("cleared", False)
            tags = ("cleared",) if is_cleared else ("pending",)
            status = "✅ 已解决" if is_cleared else "⏳ 待解决"
            tree.insert("", "end", iid=k,
                        values=(k, info2.get("pred_cls", "-"),
                                f"{info2.get('pred_conf', 0):.2f}",
                                f"{info2.get('cleared_streak', 0)}/3",
                                status, info2.get("marked_at", "-")),
                        tags=tags)

        btns = ttk.Frame(win); btns.pack(fill=tk.X, padx=10, pady=8)

        def delete_selected():
            sel = tree.selection()
            if not sel: return
            if not messagebox.askyesno("确认", f"从清单删除 {len(sel)} 条？"): return
            for k in sel: self._manual_errors.pop(k, None)
            self._save_manual_errors(); self._refresh_intensive_btn()
            win.destroy(); self.show_manual_errors()

        def reset_selected():
            sel = tree.selection()
            if not sel: return
            for k in sel:
                if k in self._manual_errors:
                    self._manual_errors[k]["cleared"] = False
                    self._manual_errors[k]["cleared_streak"] = 0
            self._save_manual_errors(); self._refresh_intensive_btn()
            win.destroy(); self.show_manual_errors()

        def clear_all():
            if not messagebox.askyesno("确认", f"清空全部 {len(self._manual_errors)} 条？"): return
            self._manual_errors.clear()
            self._save_manual_errors(); self._refresh_intensive_btn()
            win.destroy()

        ttk.Button(btns, text="删除选中", command=delete_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="重置选中", command=reset_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="清空全部", command=clear_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="关闭", command=win.destroy).pack(side=tk.RIGHT, padx=4)