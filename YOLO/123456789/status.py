# -*- coding: utf-8 -*-
"""一键状态面板：labelImg 是否在运行 + 当前标注进度 + 最近保存的标签。

用法（在 yolo_project 下）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe status.py
"""
import csv
import datetime
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
LABELS = DATASET / 'labels'
IMAGES = DATASET / 'images'
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def process_status():
    print('=== 1) 标注工具进程 ===')
    try:
        import psutil
    except ImportError:
        print('  （未安装 psutil，跳过）')
        return
    rows = []
    for p in psutil.process_iter(['pid', 'name', 'memory_info', 'num_threads', 'create_time']):
        try:
            name = (p.info['name'] or '').lower()
            if 'labelimg' in name or name.startswith('python'):
                mb = p.info['memory_info'].rss / 1024 / 1024 if p.info['memory_info'] else 0
                rows.append((p.info['pid'], p.info['name'], mb, p.info['num_threads']))
        except Exception:
            continue
    if not rows:
        print('  没有 labelImg / python 进程 —— 标注工具当前没有运行')
        print('  启动方式：双击 12_labelImg_b01_train.bat（或 10_labelImg311_train.bat）')
        return
    print('  %-8s %-12s %-10s %s' % ('PID', '进程', '内存MB', '线程'))
    for pid, name, mb, th in sorted(rows, key=lambda r: -r[2]):
        flag = '  <-- 这就是标注窗口（内存>30MB 说明已经起来了）' if mb > 30 else ''
        print('  %-8d %-12s %-10.0f %d%s' % (pid, name, mb, th, flag))


def coverage():
    print('\n=== 2) 标注覆盖率 ===')
    total_imgs = total_lbl = 0
    for split in ('train', 'val'):
        img_dir, lbl_dir = IMAGES / split, LABELS / split
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in EXTS] if img_dir.exists() else []
        lbls = [p for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'] if lbl_dir.exists() else []
        total_imgs += len(imgs)
        total_lbl += len(lbls)
        print('  [%-5s] 图片 %4d 张，有标签 %4d 张，未标注 %4d 张' %
              (split, len(imgs), len(lbls), len(imgs) - len(lbls)))
    if total_imgs:
        print('  合计：%d / %d 张有标签（%.1f%%），未标注 %d 张'
              % (total_lbl, total_imgs, 100.0 * total_lbl / total_imgs, total_imgs - total_lbl))

    print('\n  各类别实例数：')
    counts = Counter()
    per_img = Counter()
    for split in ('train', 'val'):
        for f in (LABELS / split).glob('*.txt'):
            if f.name == 'classes.txt':
                continue
            seen = set()
            for line in f.read_text(encoding='utf-8', errors='ignore').splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cid = int(float(parts[0]))
                    counts[cid] += 1
                    seen.add(cid)
            for cid in seen:
                per_img[(split, cid)] += 1
    for cid in sorted(NAMES):
        print('    %-9s 框 %4d 个，覆盖 %d 张图（train %d / val %d）'
              % (NAMES[cid], counts[cid], per_img[('train', cid)] + per_img[('val', cid)],
                 per_img[('train', cid)], per_img[('val', cid)]))


def batch_progress():
    print('\n=== 3) 分批标注进度（label_plan.csv）===')
    plan = PROJ / 'label_plan.csv'
    if not plan.exists():
        print('  没有 label_plan.csv')
        return
    stat = {}
    with plan.open(encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            b = stat.setdefault(row['batch'], [0, 0])
            b[0] += 1
            lbl = LABELS / row['split'] / (Path(row['image']).stem + '.txt')
            if lbl.exists():
                b[1] += 1
    for b in sorted(stat):
        tot, done = stat[b]
        print('  %s  %2d/%2d  %5.1f%%' % (b, done, tot, 100.0 * done / tot if tot else 0))


def recent():
    print('\n=== 4) 最近被保存/修改的标签文件 ===')
    files = []
    for split in ('train', 'val'):
        for f in (LABELS / split).glob('*.txt'):
            if f.name != 'classes.txt':
                files.append((f.stat().st_mtime, split, f.name))
    files.sort(reverse=True)
    if not files:
        print('  还没有任何标签文件')
        return
    for t, split, name in files[:8]:
        print('  %s  %s/%s' % (datetime.datetime.fromtimestamp(t).strftime('%m-%d %H:%M:%S'),
                               split, name))


def main():
    print('项目: %s\n' % PROJ)
    process_status()
    coverage()
    batch_progress()
    recent()
    print('\n提示：想看带框的预览图，先运行 _preview_boxes.py，再打开 _preview/ 目录。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
