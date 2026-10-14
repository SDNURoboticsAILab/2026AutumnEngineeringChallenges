# -*- coding: utf-8 -*-
"""标注质量审计：抓“漏标”和“误标”。

用法（在 yolo_project 下运行）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe audit_labels.py

做四件事：
  1. 漏标排查：label_worklist.csv 记录了每张图来自哪个原始类别目录（obstacle/cola/football），
     若某张图来自 cola 目录却没有任何 cola 框 —— 极可能是漏标。
  2. 误标排查：重复框（同图同类 IoU>0.85）、跨类冲突框（不同类 IoU>0.6）、
     过小框（边长<1%）、整图框（面积>60% 或四边全贴边）等典型错误。
  3. 生成待办清单：todo_no_label.txt（完全没有标签的图）、todo_obstacle.txt（缺 obstacle 框的图）。
  4. 打印汇总，便于写进报告。

输出全部用英文，避免 Windows 控制台 GBK 编码报错。
"""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
WORKLIST = PROJ / 'label_worklist.csv'
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}
SPLITS = ('train', 'val')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
MIN_SIDE = 0.01      # 归一化边长下限，比这还小基本是误框
MAX_AREA = 0.60      # 单框占整图面积上限，超过多半是把整张图框了
DUP_IOU = 0.85       # 同类重复框阈值
CONFLICT_IOU = 0.60  # 跨类冲突框阈值


def load_source_class():
    mapping = {}
    if WORKLIST.exists():
        with WORKLIST.open(encoding='utf-8-sig', newline='') as fh:
            for row in csv.DictReader(fh):
                mapping[row['image']] = row.get('source_class', '')
    return mapping


def to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = to_xyxy(*a)
    bx1, by1, bx2, by2 = to_xyxy(*b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    src_cls = load_source_class()
    problems = defaultdict(list)
    counts = Counter()
    unlabeled = []
    obstacle_todo = []

    for split in SPLITS:
        img_dir = DATASET / 'images' / split
        lbl_dir = DATASET / 'labels' / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in EXTS:
                continue
            name = img.name
            stem = img.stem
            origin = src_cls.get(name, '')
            lbl = lbl_dir / (stem + '.txt')

            if not lbl.exists():
                unlabeled.append((split, name, origin))
                if origin == 'obstacle':
                    obstacle_todo.append((split, name))
                continue

            boxes = []
            for lineno, line in enumerate(lbl.read_text(encoding='utf-8').splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    problems['bad_format'].append('%s:%d' % (name, lineno))
                    continue
                cid = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
                counts[cid] += 1
                boxes.append((cid, vals[0], vals[1], vals[2], vals[3]))

                if vals[2] < MIN_SIDE or vals[3] < MIN_SIDE:
                    problems['too_small'].append('%s:%d cls=%d w=%.4f h=%.4f'
                                                 % (name, lineno, cid, vals[2], vals[3]))
                if vals[2] * vals[3] > MAX_AREA:
                    problems['too_large'].append('%s:%d cls=%d area=%.3f'
                                                 % (name, lineno, cid, vals[2] * vals[3]))
                x1, y1, x2, y2 = to_xyxy(vals[0], vals[1], vals[2], vals[3])
                if x1 <= 0.001 and y1 <= 0.001 and x2 >= 0.999 and y2 >= 0.999:
                    problems['full_frame'].append('%s:%d cls=%d' % (name, lineno, cid))

            # 重复框 / 跨类冲突框
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    ov = iou(a[1:], b[1:])
                    if a[0] == b[0] and ov > DUP_IOU:
                        problems['duplicate_box'].append('%s cls=%d iou=%.2f' % (name, a[0], ov))
                    elif a[0] != b[0] and ov > CONFLICT_IOU:
                        problems['class_conflict'].append('%s cls=%d/%d iou=%.2f'
                                                          % (name, a[0], b[0], ov))

            # 漏标排查：来自某一类目录，却没有任何该类的框
            if origin in ('obstacle', 'cola', 'football'):
                want = {v: k for k, v in NAMES.items()}[origin]
                if want not in [b[0] for b in boxes]:
                    problems['missing_%s' % origin].append(name)
                    if origin == 'obstacle':
                        obstacle_todo.append((split, name))

    print('=' * 60)
    print('AUDIT SUMMARY')
    print('instances :', dict((NAMES[k], v) for k, v in sorted(counts.items())))
    print('no label  :', len(unlabeled))
    for key in sorted(problems):
        print('%-18s %4d' % (key, len(problems[key])))
    print('=' * 60)

    for key in ('missing_obstacle', 'missing_cola', 'missing_football', 'duplicate_box',
                'class_conflict', 'too_small', 'too_large', 'full_frame', 'bad_format'):
        if problems[key]:
            print('--- %s (showing up to 8) ---' % key)
            for item in problems[key][:8]:
                print('   ', item)

    (PROJ / 'todo_no_label.txt').write_text(
        '\n'.join('%s/%s' % (s, n) for s, n, _ in unlabeled) + '\n', encoding='utf-8')
    (PROJ / 'todo_obstacle.txt').write_text(
        '\n'.join('%s/%s' % (s, n) for s, n in obstacle_todo) + '\n', encoding='utf-8')
    print('wrote todo_no_label.txt (%d) and todo_obstacle.txt (%d)'
          % (len(unlabeled), len(obstacle_todo)))


if __name__ == '__main__':
    main()
