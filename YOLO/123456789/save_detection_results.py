# -*- coding: utf-8 -*-
"""Level 4 第 3 步：保存检测结果（完整版）。

对应考核要求（Level 4 → 实践步骤 → 3. 保存检测结果）：
    检测结果应能够显示：目标框、类别名称、置信度。
    建议至少展示：① 单个目标检测 ② 多个目标同时检测 ③ 不同场景或不同拍摄角度。

本脚本把检测结果保存成四种形式，互相印证：
    1) 画框结果图   results/saved_results/.../*.jpg          目标框 + 类别名称 + 置信度（可视化）
    2) YOLO 格式标签 results/saved_results/.../labels/*.txt    class cx cy w h conf（6 列，含置信度）
    3) 目标裁剪图   results/saved_results/.../crops/<类别名>/  按类别名分目录，每个目标一张
    4) 结构化汇总表 results/evidence_level4/05_检测结果汇总表.csv

并额外生成三张“展示情况”效果图，分别对应题目的三种建议，可直接用于报告截图。

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe save_detection_results.py
"""
import csv
import sys
from pathlib import Path

from ultralytics import YOLO

from infer_and_check import (EXTS, NAMES, PROJ, VAL_IMGS, make_montage,
                             pixel_diff, read_label, WEIGHTS)

SRC = PROJ / 'test_images'
OUT_SAVE = PROJ / 'results' / 'saved_results'
OUT_EVID = PROJ / 'results' / 'evidence_level4'
CONF = 0.25
# 用于“③ 不同场景/拍摄角度”的三张代表图（按文件名排序后取首、中、末，覆盖不同拍摄时段）
CASE3_PICK = (0, 2, 5)


def predict_save(model, sources, name):
    """对给定图片推理并把四种结果全部落盘，返回 Results 列表。"""
    return model.predict(
        source=[str(p) for p in sources],
        imgsz=640, conf=CONF,
        save=True,            # 画框结果图
        save_txt=True,        # YOLO 格式标签
        save_conf=True,       # 标签里带置信度（第 6 列）
        save_crop=True,       # 每个目标裁剪图，按类别名分目录
        show_labels=True,     # 在图上画类别名称
        show_conf=True,       # 在图上画置信度
        show_boxes=True,      # 在图上画目标框
        project=str(OUT_SAVE),
        name=name,            # 注意：必须给绝对路径，否则会被嵌套到模型记录的 runs/detect 下
        exist_ok=True,
        verbose=False,
    )


def collect(results, tag, rows):
    """把 Results 里的目标收集成行，并返回每个目标的裁剪图预期文件名。"""
    items = []
    for r in results:
        stem = Path(r.path).stem
        # 统计每个类别在当前图内出现的次序，用于拼裁剪图文件名（与 Ultralytics 命名规则一致）
        seen = {}
        for box in r.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            cx, cy, w, h = [float(v) for v in box.xywhn[0].tolist()]
            seen[cid] = seen.get(cid, 0) + 1
            suffix = '' if seen[cid] == 1 else '-%d' % seen[cid]
            rows.append({
                '图片': Path(r.path).name,
                '来源': tag,
                '类别编号': cid,
                '类别名称': NAMES[cid],
                '置信度': '%.3f' % conf,
                '框中心x': '%.4f' % cx, '框中心y': '%.4f' % cy,
                '框宽w': '%.4f' % w, '框高h': '%.4f' % h,
                '框x1': '%.4f' % (cx - w / 2), '框y1': '%.4f' % (cy - h / 2),
                '框x2': '%.4f' % (cx + w / 2), '框y2': '%.4f' % (cy + h / 2),
            })
            items.append((stem, NAMES[cid], conf, stem + suffix))
    return items


def main():
    L = []
    checks = []

    def add(s=''):
        L.append(s)

    if not WEIGHTS.exists():
        raise SystemExit('找不到权重: %s' % WEIGHTS)
    model = YOLO(str(WEIGHTS))

    new_imgs = sorted(p for p in SRC.iterdir() if p.suffix.lower() in EXTS)
    demo_imgs = [VAL_IMGS / 'image_rgb_20260728_101257.jpg',   # 单目标 obstacle
                 VAL_IMGS / 'image_rgb_20260727_163222.jpg',   # 单目标 cola
                 VAL_IMGS / 'image_rgb_20260727_103944.jpg']   # 单目标 football
    demo_imgs = [p for p in demo_imgs if p.exists()]

    add('=== Level 4 第 3 步：保存检测结果 ===')
    add('')
    add('题目要求：检测结果应能够显示「目标框」「类别名称」「置信度」；')
    add('          建议至少展示 ① 单个目标检测 ② 多个目标同时检测 ③ 不同场景/拍摄角度。')
    add('')
    add('权重: %s' % WEIGHTS)
    add('推理参数: imgsz=640, conf=%.2f, save=True, save_txt=True, save_conf=True, save_crop=True' % CONF)
    add('           show_boxes=True, show_labels=True（类别名称）, show_conf=True（置信度）')
    add('')

    # ---------------- 推理并保存 ----------------
    res_new = predict_save(model, new_imgs, 'new_images')
    res_demo = predict_save(model, demo_imgs, 'demo_single') if demo_imgs else []

    rows = []
    items_new = collect(res_new, '严格评测(新图片)', rows)
    items_demo = collect(res_demo, '展示(单目标)', rows)

    # ---------------- CSV 汇总表 ----------------
    csv_path = OUT_EVID / '05_检测结果汇总表.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['图片', '来源', '类别编号', '类别名称', '置信度',
            '框中心x', '框中心y', '框宽w', '框高h', '框x1', '框y1', '框x2', '框y2']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---------------- 四种保存形式清单 ----------------
    add('[一] 保存下来的四种结果形式')
    add('  ① 画框结果图（目标框 + 类别名称 + 置信度）')
    add('     %s' % (OUT_SAVE / 'new_images'))
    add('     %s' % (OUT_SAVE / 'demo_single'))
    add('  ② YOLO 格式标签（每行: 类别编号 cx cy w h 置信度）')
    add('     %s' % (OUT_SAVE / 'new_images' / 'labels'))
    add('     %s' % (OUT_SAVE / 'demo_single' / 'labels'))
    add('  ③ 目标裁剪图（按类别名分目录，每个目标一张）')
    add('     %s' % (OUT_SAVE / 'new_images' / 'crops'))
    add('     %s' % (OUT_SAVE / 'demo_single' / 'crops'))
    add('  ④ 结构化汇总表（可用 Excel 打开，含每个目标的类别/置信度/框坐标）')
    add('     %s' % csv_path)
    add('')

    add('[二] 完整目标清单（逐个目标：类别名称 + 置信度 + 框坐标）')
    last_img = None
    for r in list(res_new) + list(res_demo):
        fname = Path(r.path).name
        if fname != last_img:
            add('  %s' % fname)
            last_img = fname
        for box in r.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [round(float(v), 4) for v in box.xyxyn[0].tolist()]
            add('      %-9s conf=%.3f   框(归一化) x1y1x2y2 = [%.4f, %.4f, %.4f, %.4f]'
                % (NAMES[cid], conf, x1, y1, x2, y2))
    add('')

    # ---------------- 三种展示情况 ----------------
    add('[三] 三种展示情况与对应效果图')
    case1 = make_montage([(OUT_SAVE / 'demo_single' / (Path(r.path).name),
                           '① 单目标 · %s' % NAMES[int(r.boxes[0].cls[0])],
                           '%s  conf=%.3f' % (Path(r.path).stem[-15:], float(r.boxes[0].conf[0])))
                          for r in res_demo if len(r.boxes) == 1],
                         OUT_EVID / '06_情况①单目标检测.jpg', cols=3)
    add('  ① 单个目标检测: %s' % (case1 or '未生成'))

    case2 = make_montage([(OUT_SAVE / 'new_images' / (Path(r.path).name),
                           '② 多目标 · %d 个' % len(r.boxes),
                           '%s  最高conf=%.3f' % (Path(r.path).stem[-15:],
                                                  max([float(b.conf[0]) for b in r.boxes], default=0.0)))
                          for r in res_new],
                         OUT_EVID / '07_情况②多目标检测.jpg', cols=3)
    add('  ② 多个目标同时检测: %s' % (case2 or '未生成'))

    picks = [res_new[i] for i in CASE3_PICK if i < len(res_new)]
    case3_items = []
    for r in picks:
        stem = Path(r.path).stem
        add('     代表图 %s：拍摄时段 %s；真值 %d 个目标；检出 %d 个'
            % (stem, stem.replace('image_rgb_', '')[:13].replace('_', ' '),
               len(read_label(SRC / (stem + '.txt'))), len(r.boxes)))
        case3_items.append((OUT_SAVE / 'new_images' / (stem + '.jpg'),
                            '③ 场景/角度 · 检出%d个' % len(r.boxes),
                            '%s  conf最高%.3f' % (stem[-15:],
                                                  max([float(b.conf[0]) for b in r.boxes], default=0.0))))
    case3 = make_montage(case3_items, OUT_EVID / '08_情况③不同场景与角度.jpg', cols=3)
    add('  ③ 不同场景/不同拍摄角度: %s' % (case3 or '未生成'))
    add('     说明：三张代表图取自不同拍摄时段，目标在画面中的位置与尺度分布各不相同。')
    add('')

    # ---------------- 一致性校验 ----------------
    add('[四] 保存结果的一致性校验')
    # 1) txt 行数 == 检出目标数
    n_txt = n_det = 0
    for name, res in (('new_images', res_new), ('demo_single', res_demo)):
        lbl_dir = OUT_SAVE / name / 'labels'
        for r in res:
            f = lbl_dir / (Path(r.path).stem + '.txt')
            if f.exists():
                n_txt += len([x for x in f.read_text(encoding='utf-8').splitlines() if x.strip()])
            n_det += len(r.boxes)
    ok1 = n_txt == n_det and n_det > 0
    checks.append(('txt 标签行数与检出目标数一致', ok1, 'txt %d 行 vs 检出 %d 个' % (n_txt, n_det)))
    add('  [%s] 标签文件行数 = 检出目标数：%d 行 vs %d 个'
        % ('PASS' if ok1 else 'FAIL', n_txt, n_det))

    # 2) txt 里的类别与置信度逐项与预测一致
    #    注意：Ultralytics 写 txt 时置信度只保留 6 位有效数字（如 0.9824989 -> 0.982499），
    #    因此必须用容差比较，不能用精确相等。
    TOL = 1e-5
    mismatch = []
    for name, res in (('new_images', res_new), ('demo_single', res_demo)):
        lbl_dir = OUT_SAVE / name / 'labels'
        for r in res:
            f = lbl_dir / (Path(r.path).stem + '.txt')
            if not f.exists():
                mismatch.append('%s 缺少标签文件' % Path(r.path).name)
                continue
            file_rows = [x.split() for x in f.read_text(encoding='utf-8').splitlines() if x.strip()]
            pred = sorted([(int(b.cls[0]), float(b.conf[0])) for b in r.boxes])
            got = sorted([(int(float(x[0])), float(x[5])) for x in file_rows if len(x) == 6])
            bad = (len(pred) != len(got) or
                   any(p[0] != g[0] or abs(p[1] - g[1]) > TOL for p, g in zip(pred, got)))
            if bad:
                mismatch.append('%s 内容不一致' % Path(r.path).name)
    ok2 = not mismatch
    checks.append(('标签文件内容与预测一致（类别+置信度）', ok2,
                   '全部一致（容差 %.0e）' % TOL if ok2 else '; '.join(mismatch[:3])))
    add('  [%s] 标签文件内容（类别编号 + 置信度）与预测结果逐项一致'
        % ('PASS' if ok2 else 'FAIL'))
    add('       （txt 中置信度保留 6 位有效数字，故按容差 %.0e 比对）' % TOL)

    # 3) 裁剪图数量 == 检出目标数，且按类别分目录
    crop_stats, n_crop = {}, 0
    for name in ('new_images', 'demo_single'):
        d = OUT_SAVE / name / 'crops'
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                cnt = len(list(sub.glob('*.jpg')))
                crop_stats[sub.name] = crop_stats.get(sub.name, 0) + cnt
                n_crop += cnt
    ok3 = n_crop == n_det and n_det > 0
    checks.append(('裁剪图数量与检出目标数一致', ok3, '%d 张 vs %d 个' % (n_crop, n_det)))
    add('  [%s] 目标裁剪图数量 = 检出目标数：%d 张 vs %d 个' % ('PASS' if ok3 else 'FAIL', n_crop, n_det))
    add('       按类别名分目录统计：' + '，'.join('%s %d 张' % (k, v) for k, v in sorted(crop_stats.items())))

    # 4) 画框图确实画了东西
    drawn = 0
    for r in list(res_new) + list(res_demo):
        stem = Path(r.path).stem
        tag = 'new_images' if (SRC / (stem + '.jpg')).exists() else 'demo_single'
        orig = (SRC if tag == 'new_images' else VAL_IMGS) / (stem + '.jpg')
        saved = OUT_SAVE / tag / (stem + '.jpg')
        if saved.exists():
            diff, _size = pixel_diff(orig, saved)
            if diff:
                drawn += 1
    total_imgs = len(res_new) + len(res_demo)
    ok4 = drawn == total_imgs and total_imgs > 0
    checks.append(('结果图已绘制标注', ok4, '%d/%d 张与原图不同' % (drawn, total_imgs)))
    add('  [%s] 画框结果图确实绘制了内容（与原图逐像素比较）：%d/%d 张'
        % ('PASS' if ok4 else 'FAIL', drawn, total_imgs))
    add('')

    # ---------------- 结论 ----------------
    add('[五] 结论')
    for desc, ok, extra in checks:
        add('  [%s] %s（%s）' % ('PASS' if ok else 'FAIL', desc, extra))
    all_ok = all(ok for _d, ok, _e in checks)
    add('')
    add('  总结：%s' % ('检测结果已完整保存（画框图 / YOLO 标签 / 目标裁剪图 / CSV 汇总表），'
                        '三种展示情况的效果图均已生成，可直接用于报告截图。'
                        if all_ok else '存在未通过项，请先排查。'))

    OUT_EVID.mkdir(parents=True, exist_ok=True)
    (OUT_EVID / '04_保存检测结果.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    print('\n证据已写入: %s' % (OUT_EVID / '04_保存检测结果.txt'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
