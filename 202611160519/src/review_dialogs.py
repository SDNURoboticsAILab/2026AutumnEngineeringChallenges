# review_dialogs.py
# 开卷学习的弹窗：对比窗口 / 学习报告 / 人工错误清单
# v1.4:
#   - 对比窗口自动最大化、ESC 关闭、只看错误
#   - 人工引导（补漏 / 改类 / 合并）
#   - 学习规则（禁区 / 修正标签 / 漏检点）
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

        # ===== 顶部：模式切换 =====
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
        guide_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        guide_mode_var = tk.StringVar(value="off")
        for txt, val in [("关闭", "off"), ("补漏", "missed"),
                          ("改类", "wrong_cls"), ("合并", "merge")]:
            ttk.Radiobutton(guide_row, text=txt, variable=guide_mode_var,
                            value=val).pack(side=tk.LEFT, padx=6)
        guide_status = tk.StringVar(value="引导关闭")
        ttk.Label(guide_row, textvariable=guide_status,
                  font=('Consolas', 10), foreground="#cc6600").pack(side=tk.RIGHT, padx=8)

        # ===== 学习规则按钮 =====
        rule_row = ttk.LabelFrame(win, text="学习规则（禁区 / 修正标签 / 漏检点）")
        rule_row.pack(fill=tk.X, padx=8, pady=(0, 6))

        rule_stat = tk.StringVar(value="")
        ttk.Label(rule_row, textvariable=rule_stat,
                  font=('Consolas', 10), foreground="#cc0066").pack(side=tk.RIGHT, padx=8)

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
        diag_frame = ttk.LabelFrame(win, text="逐框诊断")
        diag_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        diag_text = tk.Text(diag_frame, height=8, font=('Consolas', 10),
                            bg='#f8f8f8', wrap=tk.WORD)
        diag_text.pack(fill=tk.X, padx=5, pady=5)

        # ===== 底部按钮栏 =====
        bottom = ttk.Frame(win); bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        status_var = tk.StringVar(value="提示: 点击图上任意框可选中")
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
                    "drag_start": None}
        guide_boxes = {"missed": [], "wrong_cls": [], "merge": []}

        # ===== 学习规则工具 =====
        def _refresh_rule_stat():
            rules = getattr(self.app, "learning_rules", None)
            if rules is None:
                rule_stat.set("rules 未初始化"); return
            rule_stat.set(
                f"禁区 {rules.count_forbidden()}  |  "
                f"修正 {rules.count_corrections()}  |  "
                f"漏检 {rules.count_miss_points()}")

        def _mark_forbidden():
            rules = getattr(self.app, "learning_rules", None)
            if rules is None:
                messagebox.showerror("错误", "learning_rules 未初始化"); return
            if selected["panel"] != "left":
                messagebox.showinfo("提示", "请先点击左图的一个预测框")
                return
            preds = (after_m or before_m or {}).get("_preds", [])
            if not (0 <= selected["idx"] < len(preds)):
                return
            p = preds[selected["idx"]]
            added, skipped = rules.add_forbidden_from_box(
                img_name, p["xyxy"], gt_boxes=gt_boxes,
                img_size=img.size)
            _refresh_rule_stat()
            redraw()
            if added == 0:
                messagebox.showwarning(
                    "无法设为禁区",
                    "4 个顶点都太靠近标准答案，无法生成安全禁区。\n\n"
                    "改用「✏️ 修正标签」或「❌ 标记选中框为多余」。")
            else:
                msg = f"已生成 {added} 个禁区圆"
                if skipped:
                    msg += f"（{skipped} 个顶点因太靠近 GT 被跳过）"
                status_var.set(msg)
            selected["panel"] = None; selected["idx"] = None

        def _start_correction():
            rules = getattr(self.app, "learning_rules", None)
            if rules is None:
                messagebox.showerror("错误", "learning_rules 未初始化"); return
            if selected["panel"] != "left":
                messagebox.showinfo("提示", "请先点击左图的一个预测框")
                return
            preds = (after_m or before_m or {}).get("_preds", [])
            if not (0 <= selected["idx"] < len(preds)):
                return
            old = preds[selected["idx"]]
            _open_correction_dialog(old)

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
                    cid, cls_v.get())
                _refresh_rule_stat()
                redraw()
                dlg.destroy()
                status_var.set(
                    f"✅ 已修正标签: {old_pred['cls_name']} → {cls_v.get()}")

            btns = ttk.Frame(dlg); btns.pack(pady=8)
            ttk.Button(btns, text="确定", command=ok).pack(side=tk.LEFT, padx=4)
            ttk.Button(btns, text="取消", command=dlg.destroy).pack(side=tk.LEFT, padx=4)

        ttk.Button(rule_row, text="🚫 标记为禁区",
                   command=_mark_forbidden).pack(side=tk.LEFT, padx=4)
        ttk.Button(rule_row, text="✏️ 修正标签",
                   command=_start_correction).pack(side=tk.LEFT, padx=4)
        ttk.Button(rule_row, text="📍 漏检点（拖拽补漏）",
                   command=lambda: guide_mode_var.set("missed")
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(rule_row, text="📋 查看规则",
                   command=lambda: _show_rules_dialog()
                   ).pack(side=tk.LEFT, padx=4)

        def _show_rules_dialog():
            rules = getattr(self.app, "learning_rules", None)
            if rules is None: return
            dlg = tk.Toplevel(win)
            dlg.title(f"学习规则 - {img_name}")
            dlg.geometry("700x520")
            dlg.resizable(True, True)
            dlg.bind("<Escape>", lambda e: dlg.destroy())

            ttk.Label(dlg, text=f"图片: {img_name}",
                      font=('Consolas', 11, 'bold')).pack(anchor=tk.W, padx=10, pady=6)

            fz = rules.get_forbidden_zones(img_name)
            corrs = rules.get_corrections(img_name)
            mps = rules.get_miss_points(img_name)

            ttk.Label(dlg, text=f"🚫 禁区圆: {len(fz)} 个",
                      font=('Consolas', 10, 'bold'),
                      foreground="#cc0066").pack(anchor=tk.W, padx=10, pady=(6, 0))
            txt1 = tk.Text(dlg, height=5, font=('Consolas', 9))
            txt1.pack(fill=tk.X, padx=10, pady=2)
            for z in fz:
                txt1.insert(tk.END,
                    f"  ({z['cx']:.0f},{z['cy']:.0f})  r={z['r']:.1f}\n")
            txt1.config(state=tk.DISABLED)

            ttk.Label(dlg, text=f"✏️ 标签修正: {len(corrs)} 个",
                      font=('Consolas', 10, 'bold'),
                      foreground="#cc0066").pack(anchor=tk.W, padx=10, pady=(6, 0))
            txt2 = tk.Text(dlg, height=5, font=('Consolas', 9))
            txt2.pack(fill=tk.X, padx=10, pady=2)
            for c in corrs:
                txt2.insert(tk.END,
                    f"  {c['old_xyxy']} → {c['new_xyxy']}  类别: {c['new_cls_name']}\n")
            txt2.config(state=tk.DISABLED)

            ttk.Label(dlg, text=f"📍 漏检点: {len(mps)} 个",
                      font=('Consolas', 10, 'bold'),
                      foreground="#cc0066").pack(anchor=tk.W, padx=10, pady=(6, 0))
            txt3 = tk.Text(dlg, height=5, font=('Consolas', 9))
            txt3.pack(fill=tk.X, padx=10, pady=2)
            for m in mps:
                txt3.insert(tk.END,
                    f"  {m['xyxy']}  类别: {m['cls_name']}  repeat={m['repeat']}\n")
            txt3.config(state=tk.DISABLED)

            def clear_this():
                if not messagebox.askyesno("确认", f"清空 {img_name} 的所有规则？"):
                    return
                rules.clear_forbidden(img_name)
                rules.clear_corrections(img_name)
                rules.clear_miss_points(img_name)
                _refresh_rule_stat()
                redraw()
                dlg.destroy()

            ttk.Button(dlg, text="清空本图规则",
                       command=clear_this).pack(pady=8)

        _refresh_rule_stat()

        # ===== 状态刷新 =====
        def _refresh_status():
            gm = guide_mode_var.get()
            if gm == "off":
                guide_status.set("引导关闭")
            elif gm == "missed":
                guide_status.set("补漏：在空白处拖拽画框")
            elif gm == "wrong_cls":
                guide_status.set("改类：点击预测框，然后选择新类别")
            elif gm == "merge":
                guide_status.set("合并：Shift 多选预测框，点右键合并")

            if selected["panel"] is None:
                status_var.set("提示: 点击图上任意框可选中")
            elif selected["panel"] == "left":
                preds = (after_m or before_m or {}).get("_preds", [])
                if 0 <= selected["idx"] < len(preds):
                    p = preds[selected["idx"]]
                    status_var.set(
                        f"已选: 左图预测框 #{selected['idx']} 类别={p['cls_name']} "
                        f"conf={p.get('conf', 1.0):.2f}")
            else:
                if 0 <= selected["idx"] < len(gt_boxes):
                    g = gt_boxes[selected["idx"]]
                    status_var.set(
                        f"已选: 右图 GT 框 #{selected['idx']} 类别={g['cls_name']}")

        # ===== 命中测试 / 坐标转换 =====
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

        # ===== 点击 / 拖拽 =====
        def _on_left_click(event):
            gm = guide_mode_var.get()
            if gm == "missed":
                selected["drag_start"] = _canvas_to_img(left_canvas, event)
                return
            preds = (after_m or before_m or {}).get("_preds", [])
            hit = _hit_test(left_canvas, event, preds)
            if gm == "wrong_cls" and hit is not None:
                _do_wrong_cls(hit)
                return
            if gm == "merge" and hit is not None:
                is_shift = bool(event.state & SHIFT_MASK)
                if is_shift:
                    if hit in selected["indices_left"]:
                        selected["indices_left"].remove(hit)
                    else:
                        selected["indices_left"].append(hit)
                else:
                    selected["indices_left"] = [hit]
                selected["panel"] = "left"
                selected["idx"] = hit
                _refresh_status()
                redraw()
                return
            selected["panel"] = "left" if hit is not None else None
            selected["idx"] = hit
            if hit is None:
                selected["indices_left"] = []
            _refresh_status()
            redraw()

        def _on_right_click(event):
            gm = guide_mode_var.get()
            if gm == "merge" and len(selected["indices_left"]) >= 2:
                _do_merge()
                return
            hit = _hit_test(right_canvas, event, gt_boxes)
            selected["panel"] = "right" if hit is not None else None
            selected["idx"] = hit
            _refresh_status()
            redraw()

        def _on_drag(event):
            if guide_mode_var.get() != "missed":
                return
            if selected.get("drag_start") is None:
                return
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
            if guide_mode_var.get() != "missed":
                return
            if selected.get("drag_start") is None:
                return
            x1, y1 = selected["drag_start"]
            x2, y2 = _canvas_to_img(left_canvas, event)
            selected["drag_start"] = None
            left_canvas.delete("drag_rect")
            if x2 is None or abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                return
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            _do_missed([x1, y1, x2, y2])

        # ===== 补漏 =====
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
                # 写 guidance
                try:
                    self.app.guidance.add_missed(img_name, xyxy, cid, v.get())
                except Exception as e:
                    print(f"[guidance 保存失败] {e}")
                # 写 learning_rules
                rules = getattr(self.app, "learning_rules", None)
                if rules is not None:
                    try:
                        rules.add_miss_point(img_name, xyxy, cid, v.get())
                    except Exception as e:
                        print(f"[learning_rules 保存失败] {e}")
                _refresh_rule_stat()
                cls_win.destroy()
                _refresh_status()
                redraw()

            ttk.Button(cls_win, text="确定", command=ok).pack(pady=6)

        def _do_wrong_cls(pred_idx):
            preds = (after_m or before_m or {}).get("_preds", [])
            if not (0 <= pred_idx < len(preds)): return
            p = preds[pred_idx]
            cls_win = tk.Toplevel(win)
            cls_win.title("改类")
            cls_win.geometry("300x180")
            cls_win.bind("<Escape>", lambda e: cls_win.destroy())
            ttk.Label(cls_win, text=f"当前类别: {p['cls_name']}\n改为:").pack(pady=8)
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
                cls_win.destroy()
                _refresh_status()
                redraw()

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
            _refresh_status()
            redraw()

        left_canvas.bind("<Button-1>", _on_left_click)
        left_canvas.bind("<B1-Motion>", _on_drag)
        left_canvas.bind("<ButtonRelease-1>", _on_release)
        left_canvas.bind("<Button-3>", _on_right_click)
        right_canvas.bind("<Button-1>", _on_right_click)

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
                    f"把左侧预测框标记为「多余」?\n\n类别: {p['cls_name']}\n"
                    f"→ 加入人工错误清单，触发高强度训练"):
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
                if not show_all and is_matched:
                    continue
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
                if not show_all and is_matched:
                    continue
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

            # 引导框（紫色虚线）
            for ms in guide_boxes["missed"]:
                x1, y1, x2, y2 = ms["xyxy"]
                for seg in [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]:
                    draw.line(seg, fill='#cc00ff', width=3)

            # 禁区圆（红色虚线圆）
            rules = getattr(self.app, "learning_rules", None)
            if rules is not None and panel_name == "left":
                for z in rules.get_forbidden_zones(img_name):
                    cx, cy, r = z["cx"], z["cy"], z["r"]
                    # 用多边形近似圆
                    pts = []
                    import math
                    for k in range(24):
                        a = k * math.pi / 12
                        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                    for k in range(len(pts)):
                        a = pts[k]; b = pts[(k + 1) % len(pts)]
                        draw.line([a, b], fill='#ff0000', width=2)

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
            for info in m.get("required_info", []):
                gi, cls, status = info
                tag_map = {"hit": "命中", "cls_wrong": "类别错", "missed": "漏检"}
                text_widget.insert(tk.END,
                    f"  ★ 必要框 GT#{gi} {cls}: {tag_map.get(status, status)}\n")
            gt_boxes_local = m.get("_gt_boxes", gt_boxes)
            for gi in m.get("missed", []):
                if gi < len(gt_boxes_local):
                    g = gt_boxes_local[gi]
                    text_widget.insert(tk.END,
                        f"  ✗ 漏检: GT#{gi} {g['cls_name']}\n")
            for pj in m.get("extra", []):
                text_widget.insert(tk.END, f"  ✗ 误检: 预测#{pj}\n")
            for pj in m.get("duplicate_extra", []):
                text_widget.insert(tk.END, f"  ⚠ 重复预测: 预测#{pj}\n")
            for item in m.get("cls_error", []):
                gi, pj, iou, g_cls, p_cls = item
                text_widget.insert(tk.END,
                    f"  ✗ 类别错: GT#{gi} {g_cls} → {p_cls}  (IoU={iou:.2f})\n")

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

    # ==================== 人工错误清单弹窗 ====================
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