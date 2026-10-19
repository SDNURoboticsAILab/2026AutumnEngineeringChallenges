# -*- coding: utf-8 -*-
"""Level 4 第 2 步：对训练集之外的新图片进行推理（完整版）。

做四件事：
  1) 加载自己训练的权重（runs/detect/train_holdout/weights/best.pt）。
     这个模型训练时已把 test_images/ 里的 6 张图移出数据集，所以它们是真·没见过的新图片。
  2) 对 6 张新图片推理，结果图（含目标框 + 类别名称 + 置信度）保存到 results/predict_holdout/。
  3) 与随图带出的真值标签比对（IoU>=0.5 记为命中），输出每类 TP/FP/FN 与精确率/召回率。
  4) 补齐 Level 4 第 3 步建议的三种展示情况，并生成一张拼接效果图：
       ① 单个目标检测   ② 多个目标同时检测   ③ 不同场景 / 不同拍摄角度
     ①的素材取自 dataset 的 val 划分（不在 train 集内、模型训练时未见过其权重更新，
     但参与过验证，因此只作展示、不计入严格评测指标）。

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe infer_and_check.py
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe infer_and_check.py --conf 0.15
"""
import argparse
import math
import sys
from pathlib import Path

from ultralytics import YOLO

PROJ = Path(__file__).resolve().parent
WEIGHTS = PROJ / 'runs' / 'detect' / 'train_holdout' / 'weights' / 'best.pt'
SRC = PROJ / 'test_images'                       # 严格评测：真·新图片
VAL_IMGS = PROJ / 'dataset' / 'images' / 'val'   # 展示用素材来源（val 划分）
VAL_LBLS = PROJ / 'dataset' / 'labels' / 'val'
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
FONT = 'C:/Windows/Fonts/msyh.ttc'
OUT_EVID = PROJ / 'results' / 'evidence_level4'


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def xywh2xyxy(x, y, w, h):
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def read_label(path):
    """读一个 YOLO 标签文件，返回 [(类别, x1, y1, x2, y2), ...]（归一化坐标）。"""
    out = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            p = line.split()
            if len(p) == 5:
                out.append((int(float(p[0])),) + xywh2xyxy(*[float(v) for v in p[1:]]))
    return out


def single_object_candidates():
    """找出 val 划分中“只有一个目标”的图片，按目标面积从大到小排列。

    只用于展示“单个目标检测”这一情况。
    """
    cand = {c: [] for c in NAMES}
    if not VAL_LBLS.exists():
        return cand
    for f in sorted(VAL_LBLS.glob('*.txt')):
        if f.name == 'classes.txt':
            continue
        gt = read_label(f)
        if len(gt) != 1:
            continue
        img = VAL_IMGS / (f.stem + '.jpg')
        if not img.exists():
            continue
        cid = gt[0][0]
        _, x1, y1, x2, y2 = gt[0]
        cand[cid].append(((x2 - x1) * (y2 - y1), f.stem))
    for c in cand:
        cand[c].sort(reverse=True)
    return cand


def pick_demo(model, conf, budget=5):
    """为每个类别挑一张单目标展示图，且要求模型在该图上恰好检出 1 个同类目标。"""
    cand = single_object_candidates()
    picked, tried = {}, {}
    for cid in sorted(NAMES):
        tried[cid] = []
        for _area, stem in cand[cid][:budget]:
            img = VAL_IMGS / (stem + '.jpg')
            r = model.predict(source=str(img), imgsz=640, conf=conf, verbose=False)[0]
            tried[cid].append((stem, len(r.boxes)))
            if len(r.boxes) == 1 and int(r.boxes[0].cls[0]) == cid:
                picked[cid] = stem
                break
    return picked, tried, cand


def pixel_diff(orig, drawn):
    """比较原图与结果图的像素差异，用来证明结果图上确实画了东西。

    用 tobytes() 直接比较字节（等价于逐像素比较），比 getdata() 快且不触发弃用警告。
    """
    from PIL import Image
    try:
        a = Image.open(orig).convert('RGB')
        b = Image.open(drawn).convert('RGB')
    except Exception:
        return None, None
    if a.size != b.size:
        return None, a.size
    ba, bb = a.tobytes(), b.tobytes()
    step = 3  # 每个像素 3 字节
    diff = sum(1 for i in range(0, len(ba), step) if ba[i:i + step] != bb[i:i + step])
    return diff, a.size


def make_montage(items, out_path, cols=3, cell_w=430, caption_h=58, pad=12):
    """把若干张结果图拼成一张效果图，每格下面写两行说明。

    items: [(图片路径, 第一行说明, 第二行说明), ...]
    """
    from PIL import Image, ImageDraw, ImageFont
    if not items:
        return None
    font = ImageFont.truetype(FONT, 17)
    rows = math.ceil(len(items) / cols)
    thumbs = []
    for path, l1, l2 in items:
        im = Image.open(path).convert('RGB')
        ratio = cell_w / im.width
        im = im.resize((cell_w, max(1, round(im.height * ratio))))
        thumbs.append((im, l1, l2))
    cell_h = max(t[0].height for t in thumbs) + caption_h
    W = cols * cell_w + (cols + 1) * pad
    H = rows * cell_h + (rows + 1) * pad
    canvas = Image.new('RGB', (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for i, (im, l1, l2) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        canvas.paste(im, (x, y))
        draw.rectangle([x, y, x + cell_w - 1, y + im.height], outline=(180, 180, 180))
        draw.text((x + 2, y + im.height + 6), l1, fill=(20, 20, 20), font=font)
        draw.text((x + 2, y + im.height + 30), l2, fill=(90, 90, 90), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--weights', default=str(WEIGHTS))
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit('找不到权重: %s' % weights)

    model = YOLO(str(weights))
    imgs = sorted(p for p in SRC.iterdir() if p.suffix.lower() in EXTS)
    if not imgs:
        raise SystemExit('test_images/ 里没有图片')

    lines = []
    lines.append('=== Level 4 第 2 步：对训练集之外的新图片进行推理 ===')
    lines.append('')
    lines.append('[一] 推理设置')
    lines.append('  权重文件  : %s' % weights)
    lines.append('  为什么是新图片: 该模型训练时已把这 %d 张图移出数据集（见 test_images/_manifest.txt），'
                 % len(imgs))
    lines.append('                  因此模型在训练与验证中都没有见过它们。')
    lines.append('  推理图片  : %s' % SRC)
    lines.append('  推理参数  : imgsz=640, conf=%.2f, save=True' % args.conf)
    lines.append('              show_labels=True（画类别名）, show_conf=True（画置信度）, show_boxes=True')
    lines.append('  结果目录  : %s' % (PROJ / 'results' / 'predict_holdout'))
    lines.append('')

    # ---------------- [二] 严格评测：6 张真·新图片 ----------------
    lines.append('[二] 严格评测：%d 张从未参与训练与验证的新图片' % len(imgs))
    lines.append('')
    results = model.predict(source=[str(p) for p in imgs], imgsz=640, conf=args.conf,
                            save=True, project=str(PROJ / 'results'),
                            name='predict_holdout', exist_ok=True,
                            show_labels=True, show_conf=True, show_boxes=True, verbose=False)

    stat = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in NAMES}
    det_summary = {}       # stem -> (检出目标数, 最高置信度, 类别计数文本)
    for r in results:
        stem = Path(r.path).stem
        gt = read_label(SRC / (stem + '.txt'))
        preds = []
        for box in r.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            x, y, w, h = box.xywhn[0].tolist()
            preds.append((cid, conf, xywh2xyxy(x, y, w, h)))
        preds.sort(key=lambda t: -t[1])

        counts = {}
        for cid, _c, _b in preds:
            counts[NAMES[cid]] = counts.get(NAMES[cid], 0) + 1
        det_summary[stem] = (len(preds),
                             max([c for _i, c, _b in preds], default=0.0),
                             '、'.join('%s×%d' % (k, v) for k, v in counts.items()) or '无')

        matched_gt = set()
        lines.append('%s' % Path(r.path).name)
        lines.append('  真值: ' + (', '.join(NAMES[c] for c, *_ in gt) or '（无）'))
        if not preds:
            lines.append('  预测: （未检出目标）')
        for cid, conf, box in preds:
            best_i, best_iou = -1, 0.0
            for i, (gc, *gbox) in enumerate(gt):
                if i in matched_gt or gc != cid:
                    continue
                v = iou(box, gbox)
                if v > best_iou:
                    best_i, best_iou = i, v
            px = [round(v * 100) / 100 for v in box]
            if best_i >= 0 and best_iou >= 0.5:
                matched_gt.add(best_i)
                stat[cid]['tp'] += 1
                lines.append('  预测: %-9s conf=%.3f  框(归一化)x1y1x2y2=%s  -> 命中真值 (IoU=%.2f)'
                             % (NAMES[cid], conf, px, best_iou))
            else:
                stat[cid]['fp'] += 1
                lines.append('  预测: %-9s conf=%.3f  框(归一化)x1y1x2y2=%s  -> 无对应真值（误报，最佳IoU=%.2f）'
                             % (NAMES[cid], conf, px, best_iou))
        for i, (gc, *_rest) in enumerate(gt):
            if i not in matched_gt:
                stat[gc]['fn'] += 1
                lines.append('  漏检: %s（真值里有，模型没报出来）' % NAMES[gc])
        lines.append('')

    lines.append('-' * 76)
    lines.append('%-10s %5s %5s %5s %8s %8s' % ('类别', 'TP', 'FP', 'FN', '精确率', '召回率'))
    tp_all = fp_all = fn_all = 0
    rows_txt = []
    for c, s in stat.items():
        prec = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) else 0.0
        rec = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) else 0.0
        rows_txt.append('%-10s %5d %5d %5d %8.3f %8.3f' % (NAMES[c], s['tp'], s['fp'], s['fn'], prec, rec))
        tp_all += s['tp']; fp_all += s['fp']; fn_all += s['fn']
    prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
    rec = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0
    lines += rows_txt
    lines.append('%-10s %5d %5d %5d %8.3f %8.3f' % ('合计', tp_all, fp_all, fn_all, prec, rec))
    lines.append('  判定规则: 预测框与真值框 IoU>=0.5 且类别相同记为命中(TP)')
    lines.append('  置信度阈值 conf=%.2f（越低召回越高、误报也越多）' % args.conf)
    lines.append('')

    # ---------------- [三] 三种展示情况 ----------------
    lines.append('[三] 三种展示情况（对应 Level 4 第 3 步的建议）')
    lines.append('')
    picked, tried, cand = pick_demo(model, args.conf)

    montage_items = []
    # ① 单个目标
    lines.append('① 单个目标检测（模型在图中只检出 1 个目标）')
    if picked:
        demo_results = model.predict(source=[str(VAL_IMGS / (picked[c] + '.jpg')) for c in sorted(picked)],
                                     imgsz=640, conf=args.conf, save=True,
                                     project=str(PROJ / 'results'), name='predict_demo',
                                     exist_ok=True, show_labels=True, show_conf=True,
                                     show_boxes=True, verbose=False)
        by_stem = {Path(r.path).stem: r for r in demo_results}
        for c in sorted(picked):
            stem = picked[c]
            r = by_stem.get(stem)
            n = len(r.boxes) if r is not None else 0
            cf = float(r.boxes[0].conf[0]) if (r is not None and n) else 0.0
            lines.append('   %-9s : %s.jpg   检出 %d 个，conf=%.3f'
                         % (NAMES[c], stem, n, cf))
            saved = PROJ / 'results' / 'predict_demo' / (stem + '.jpg')
            if saved.exists():
                montage_items.append((saved, '① 单目标 · %s' % NAMES[c],
                                      '%s  conf=%.3f' % (stem[-15:], cf)))
        for c in sorted(NAMES):
            if c not in picked:
                lines.append('   %-9s : 未找到满足条件的示例（候选 %d 张，已试 %d 张：%s）'
                             % (NAMES[c], len(cand[c]), len(tried.get(c, [])),
                                ', '.join('%s→检出%d' % t for t in tried.get(c, [])) or '无候选'))
        lines.append('   说明: 展示素材取自 dataset/images/val（不在 train 集内，权重未在它们身上更新过），')
        lines.append('         但它们参与过验证，因此只作展示，不计入上面 [二] 的严格评测指标。')
    else:
        lines.append('   未找到可用的单目标示例')
    lines.append('')

    # ② 多个目标 + ③ 不同场景/角度（用 6 张严格评测图）
    lines.append('② 多个目标同时检测（同一张图内检出 2 个及以上目标）')
    multi = [p for p in imgs if det_summary.get(p.stem, (0,))[0] >= 2]
    for p in multi:
        n, cf, cnt = det_summary[p.stem]
        lines.append('   %s.jpg   检出 %d 个：%s（最高 conf=%.3f）' % (p.stem, n, cnt, cf))
        saved = PROJ / 'results' / 'predict_holdout' / (p.stem + '.jpg')
        if saved.exists():
            montage_items.append((saved, '② 多目标 · %d 个' % n, '%s  最高conf=%.3f' % (p.stem[-15:], cf)))
    lines.append('')

    lines.append('③ 不同场景 / 不同拍摄角度（按拍摄时段与目标位置分布说明，均由数据得出）')
    for p in imgs:
        gt = read_label(SRC / (p.stem + '.txt'))
        if not gt:
            continue
        cxs = [round((g[1] + g[3]) / 2, 2) for g in gt]
        areas = [round((g[3] - g[1]) * (g[4] - g[2]), 4) for g in gt]
        # 拍摄时段由文件名中的日期_时间给出
        parts = p.stem.replace('image_rgb_', '')
        when = '%s %s:%s' % (parts[0:8], parts[9:11], parts[11:13]) if len(parts) >= 13 else p.stem
        spread = '画面左' if max(cxs) <= 0.4 else ('画面右' if min(cxs) >= 0.6 else '横跨画面中部')
        lines.append('   %s.jpg  拍摄时段 %s  目标数 %d  中心横坐标 %s（%s）  目标面积 %s'
                     % (p.stem, when, len(gt), cxs, spread, areas))
    if len({p.stem[9:17] for p in imgs}) > 1:
        lines.append('   → 6 张图来自多个拍摄时段，目标位置与尺度分布各不相同。')
    lines.append('')

    # ---------------- [四] 结果图与效果图 ----------------
    lines.append('[四] 检测结果保存与效果图')
    lines.append('  结果图目录（每张都画了目标框 + 类别名称 + 置信度）:')
    lines.append('    %s' % (PROJ / 'results' / 'predict_holdout'))
    lines.append('    %s （单个目标展示用）' % (PROJ / 'results' / 'predict_demo'))
    lines.append('')
    lines.append('  画框是否真的画上去了（逐像素对比原图与结果图）:')
    drawn_ok = 0
    for p in imgs:
        saved = PROJ / 'results' / 'predict_holdout' / (p.stem + '.jpg')
        if not saved.exists():
            lines.append('    %-34s 结果图缺失' % p.name)
            continue
        diff, size = pixel_diff(p, saved)
        if diff is None:
            lines.append('    %-34s 尺寸不一致，无法比较' % p.name)
        else:
            drawn_ok += 1 if diff > 0 else 0
            lines.append('    %-34s 差异像素 %6d / 共 %d（%s）'
                         % (p.name, diff, size[0] * size[1],
                            '已绘制标注' if diff > 0 else '⚠️ 与原图完全一致，未绘制'))
    lines.append('    → %d/%d 张结果图与原图不同，说明标注确实绘制在图上。' % (drawn_ok, len(imgs)))
    lines.append('')

    montage_path = make_montage(montage_items, OUT_EVID / '03_检测效果图_拼接.jpg')
    if montage_path:
        lines.append('  拼接效果图（可直接放进报告）: %s' % montage_path)
        lines.append('    共 %d 格：① 单目标 %d 张；② 多目标 %d 张'
                     % (len(montage_items), len(picked), len(multi)))
    lines.append('')

    txt = '\n'.join(lines)
    OUT_EVID.mkdir(parents=True, exist_ok=True)
    (OUT_EVID / '02_对新图片推理.txt').write_text(txt + '\n', encoding='utf-8')
    print(txt)
    print('\n证据已写入: %s' % (OUT_EVID / '02_对新图片推理.txt'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
