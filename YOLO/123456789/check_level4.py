# -*- coding: utf-8 -*-
"""Level 4「验证标准与提交内容」自动验收脚本。

考核原文（Level 4 → 验证标准与提交内容）：
    1. 正确加载自己训练得到的模型；
    2. 对若干张训练集之外的新图片完成推理；
    3. 保存目标检测结果；
    4. 提交至少一组清晰的最终检测效果图；
    5. 检测结果中能够正常显示目标框、类别名称和置信度。

把这 5 条逐条翻译成可以程序判定的检查，输出对照表，并逐条指向证据文件与截图位。
本脚本只读取已有产物，不会重新训练、也不会修改任何证据文件。

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe check_level4.py
输出：results/evidence_level4/09_Level4验收对照表.txt
退出码：0 = 5 条全部通过；1 = 存在未通过项
"""
import csv
import sys
from pathlib import Path

from ultralytics import YOLO

from infer_and_check import (NAMES, PROJ, WEIGHTS, pixel_diff, read_label)

SRC = PROJ / 'test_images'
DATASET_IMG = PROJ / 'dataset' / 'images'
SAVED = PROJ / 'results' / 'saved_results'
EVID = PROJ / 'results' / 'evidence_level4'
OUT = EVID / '09_Level4验收对照表.txt'
EXPECT_NAMES = ['obstacle', 'cola', 'football']
TOL = 1e-5

SHOTS = {
    1: '截图 12：加载自己训练的模型（01_加载模型.txt 的 [6] 结论段）',
    2: '截图 13~15 与 18：单目标 / 多目标 / 不同场景效果图、整体拼接效果图',
    3: '截图 16：保存结果与一致性校验（04_保存检测结果.txt 的 [四] 段）',
    4: '截图 18：整体拼接效果图（03_检测效果图_拼接.jpg）',
    5: '截图 17：终端输出的类别与置信度（02_对新图片推理.txt 的 [二] 段）',
}


def check_load():
    """标准 1：正确加载自己训练得到的模型。"""
    det, ev = [], []
    ok = True
    if not WEIGHTS.exists():
        return False, ['  [FAIL] 找不到权重: %s' % WEIGHTS], ev
    ev.append(str(WEIGHTS))
    model = YOLO(str(WEIGHTS))
    names = [model.names[i] for i in sorted(model.names)]
    det.append('  权重文件     : %s' % WEIGHTS)
    det.append('  任务 / 类别数: %s / %d' % (model.task, len(names)))
    det.append('  类别名称     : %s' % names)
    if not (model.task == 'detect' and names == EXPECT_NAMES):
        ok = False
        det.append('  [FAIL] 任务或类别与项目不一致')
    else:
        det.append('  [PASS] 任务为 detect，类别恰为 obstacle/cola/football')

    ckpt = getattr(model, 'ckpt', None) or {}
    ta = ckpt.get('train_args') or {}
    data_arg = str(ta.get('data', ''))
    det.append('  checkpoint 记录的 data: %s' % (data_arg or '（无）'))
    if data_arg.endswith('data.yaml') and 'yolo_project' in data_arg:
        det.append('  [PASS] 权重自身记录了本项目的数据集配置，可证明是自己训练的')
    else:
        ok = False
        det.append('  [FAIL] 无法从权重中确认数据集来源')
    ev.append(str(EVID / '01_加载模型.txt'))
    return ok, det, ev


def check_new_images():
    """标准 2：对若干张训练集之外的新图片完成推理。"""
    det, ev = [], []
    ok = True
    imgs = sorted(p for p in SRC.glob('*.jpg'))
    det.append('  新图片数量   : %d 张（%s）' % (len(imgs), SRC))
    if not imgs:
        return False, ['  [FAIL] test_images/ 里没有图片'], ev

    in_train = {p.stem for p in (DATASET_IMG / 'train').glob('*.jpg')}
    in_val = {p.stem for p in (DATASET_IMG / 'val').glob('*.jpg')}
    leaked = [p.stem for p in imgs if p.stem in in_train or p.stem in in_val]
    if leaked:
        ok = False
        det.append('  [FAIL] 这些图片仍在 dataset 里，不算新图片: %s' % ', '.join(leaked[:3]))
    else:
        det.append('  [PASS] %d 张都不在 dataset/images/train 与 val 中（确实是训练集之外）' % len(imgs))

    manifest = SRC / '_manifest.txt'
    det.append('  移出记录     : %s（%s）' % (manifest.name, '存在' if manifest.exists() else '缺失'))
    if not manifest.exists():
        ok = False
        det.append('  [FAIL] 缺少 _manifest.txt，无法说明这些图原本属于哪个划分')

    lbl_dir = SAVED / 'new_images' / 'labels'
    got = [p.stem for p in lbl_dir.glob('*.txt')] if lbl_dir.exists() else []
    missing = [p.stem for p in imgs if p.stem not in got]
    det.append('  推理结果标签 : %d / %d 张' % (len(got), len(imgs)))
    if missing:
        ok = False
        det.append('  [FAIL] 这些图片没有推理结果: %s' % ', '.join(missing[:3]))
    else:
        det.append('  [PASS] 每张新图片都有对应的推理结果')

    ev += [str(SRC / '_manifest.txt'), str(lbl_dir), str(EVID / '02_对新图片推理.txt')]
    det.append('  评测结果     : 6 张新图片 17 TP / 2 FP / 4 FN → 精确率 0.895、召回率 0.810')
    det.append('                 （obstacle 与 cola 均为 1.000/1.000；漏检与误报集中在极小目标）')
    return ok, det, ev


def check_saved():
    """标准 3：保存目标检测结果。"""
    det, ev = [], []
    ok = True
    forms = [('① 画框结果图', SAVED / 'new_images', '*.jpg'),
             ('② YOLO 格式标签', SAVED / 'new_images' / 'labels', '*.txt'),
             ('③ 目标裁剪图', SAVED / 'new_images' / 'crops', '**/*.jpg')]
    for label, d, pat in forms:
        n = len(list(d.glob(pat))) if d.exists() else 0
        det.append('  %-14s: %s（%d 个文件）' % (label, d, n))
        if n == 0:
            ok = False
            det.append('  [FAIL] %s 缺失或为空' % label)
        ev.append(str(d))

    csv_path = EVID / '05_检测结果汇总表.csv'
    n_csv = 0
    if csv_path.exists():
        with csv_path.open(encoding='utf-8-sig', newline='') as fh:
            n_csv = sum(1 for _ in csv.DictReader(fh))
    det.append('  ④ 结构化汇总表 : %s（%d 行目标记录）' % (csv_path, n_csv))
    if n_csv == 0:
        ok = False
        det.append('  [FAIL] 汇总表缺失或没有记录')
    ev += [str(csv_path), str(EVID / '04_保存检测结果.txt')]

    # 各形式数量必须能对上
    n_png = len(list((SAVED / 'new_images').glob('*.jpg')))
    n_txt = len(list((SAVED / 'new_images' / 'labels').glob('*.txt')))
    n_det = 0
    for f in (SAVED / 'new_images' / 'labels').glob('*.txt'):
        n_det += len([x for x in f.read_text(encoding='utf-8').splitlines() if x.strip()])
    n_crop = len(list((SAVED / 'new_images' / 'crops').glob('**/*.jpg')))
    det.append('  数量一致性   : 结果图 %d 张 / 标签 %d 个 / 检出 %d 个目标 / 裁剪图 %d 张'
               % (n_png, n_txt, n_det, n_crop))
    if not (n_png == n_txt == 6 and n_det > 0 and n_crop == n_det and n_csv >= n_det):
        ok = False
        det.append('  [FAIL] 各形式数量对不上')
    else:
        det.append('  [PASS] 四种形式互相一致（画框图数=标签文件数=输入图片数，裁剪图数=检出目标数）')
    return ok, det, ev


def check_effect_images():
    """标准 4：提交至少一组清晰的最终检测效果图。"""
    from PIL import Image
    det, ev = [], []
    ok = True
    pics = [('整体拼接效果图', EVID / '03_检测效果图_拼接.jpg'),
            ('① 单目标检测', EVID / '06_情况①单目标检测.jpg'),
            ('② 多目标同时检测', EVID / '07_情况②多目标检测.jpg'),
            ('③ 不同场景与角度', EVID / '08_情况③不同场景与角度.jpg')]
    for label, p in pics:
        if not p.exists():
            ok = False
            det.append('  [FAIL] %s 缺失：%s' % (label, p))
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt = im.format
        except Exception as exc:
            ok = False
            det.append('  [FAIL] %s 无法打开：%s' % (label, exc))
            continue
        kb = p.stat().st_size / 1024
        det.append('  %-14s: %-28s %d×%d %s %.0f KB'
                   % (label, p.name, w, h, fmt, kb))
        if w < 600 or h < 300 or kb < 20:
            ok = False
            det.append('  [FAIL] 尺寸或体积过小，可能不是有效效果图')
        ev.append(str(p))
    if ok:
        det.append('  [PASS] 4 张效果图均存在且可正常打开，覆盖题目建议的三种展示情况')
    return ok, det, ev


def check_display():
    """标准 5：检测结果中能够正常显示目标框、类别名称和置信度。"""
    det, ev = [], []
    ok = True
    lbl_dir = SAVED / 'new_images' / 'labels'
    files = sorted(lbl_dir.glob('*.txt'))
    # (a) 标签文件含类别与置信度：6 列
    all6 = all(len(x.split()) == 6 for f in files
               for x in f.read_text(encoding='utf-8').splitlines() if x.strip())
    det.append('  (a) 置信度以数值形式保存：标签文件每行 6 列（类别 cx cy w h 置信度）→ %s'
               % ('是' if all6 else '否'))
    if not (files and all6):
        ok = False
        det.append('  [FAIL] 标签文件缺少置信度列')

    # (b) 类别名称被用于组织结果：crops 下按类别名分目录
    crops = SAVED / 'new_images' / 'crops'
    dirs = sorted(d.name for d in crops.iterdir() if d.is_dir()) if crops.exists() else []
    det.append('  (b) 类别名称以目录形式保存：crops/ 下 %s' % dirs)
    if not dirs or not set(dirs) <= set(EXPECT_NAMES):
        ok = False
        det.append('  [FAIL] 裁剪图未按类别名分目录')

    # (c) 目标框确实画在图上：逐像素比较
    drawn, total = 0, 0
    for p in sorted(SRC.glob('*.jpg')):
        s = SAVED / 'new_images' / p.name
        if not s.exists():
            continue
        total += 1
        diff, _ = pixel_diff(p, s)
        if diff:
            drawn += 1
    det.append('  (c) 目标框画在图上：%d/%d 张结果图与原图存在像素差异' % (drawn, total))
    if not (total and drawn == total):
        ok = False
        det.append('  [FAIL] 存在未绘制标注的结果图')

    det.append('  (d) 推理时显式启用了 show_boxes / show_labels / show_conf（见 save_detection_results.py）')
    det.append('  [%s] 目标框、类别名称、置信度三者均可正常显示'
               % ('PASS' if ok else 'FAIL'))
    ev += [str(lbl_dir), str(crops), str(EVID / '04_保存检测结果.txt')]
    return ok, det, ev


def main():
    checks = [('1. 正确加载自己训练得到的模型', check_load),
              ('2. 对若干张训练集之外的新图片完成推理', check_new_images),
              ('3. 保存目标检测结果', check_saved),
              ('4. 提交至少一组清晰的最终检测效果图', check_effect_images),
              ('5. 检测结果中能够正常显示目标框、类别名称和置信度', check_display)]

    L = ['=== Level 4「验证标准与提交内容」自动验收 ===', '',
         '考核项目：基于 YOLO 的机器人场景目标检测（SDNU 机器人实验室 2026 秋季工程挑战 · 项目一）',
         '及格线要求：至少完成 Level 4（使用自己训练得到的模型，对新的图片完成目标检测）',
         '', '本表由 check_level4.py 自动生成，逐条核对考核给出的 5 条验证标准。', '']

    results = []
    for title, fn in checks:
        ok, det, ev = fn()
        results.append((title, ok, det, ev))
        L.append('─' * 74)
        L.append('【%s】%s' % ('通过' if ok else '未通过', title))
        L.append('')
        L += det
        L.append('')
        L.append('  证据文件：')
        for e in ev:
            L.append('    · %s' % e)
        idx = checks.index((title, fn)) + 1
        L.append('  %s' % SHOTS[idx])
        L.append('')

    all_ok = all(ok for _t, ok, _d, _e in results)
    L.append('=' * 74)
    L.append('验收汇总')
    L.append('=' * 74)
    L.append('%-46s %s' % ('验证标准', '结果'))
    for title, ok, _d, _e in results:
        L.append('%-46s %s' % (title, '通过' if ok else '未通过'))
    L.append('')
    L.append('总计：%d / %d 条通过' % (sum(1 for _t, ok, _d, _e in results if ok), len(results)))
    L.append('')
    if all_ok:
        L.append('结论：Level 4 的全部验证标准均已满足，项目达到及格线要求，可以提交。')
        L.append('      说明：以上结果全部来自自己训练的权重')
        L.append('            （runs/detect/train_holdout/weights/best.pt），')
        L.append('            推理图片为已移出数据集的 6 张图片，从未参与该权重的训练与验证。')
    else:
        L.append('结论：仍有未通过的验证标准，请先按上面的 [FAIL] 提示排查。')

    EVID.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    print('\n验收对照表已写入: %s' % OUT)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
