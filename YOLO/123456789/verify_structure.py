# -*- coding: utf-8 -*-
"""目录结构合规检查：逐条核对考核 README 要求的目录结构。

用法（在 yolo_project 下）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe verify_structure.py

核对两组要求：
  【Level 1 推荐目录结构】dataset/images/{train,val}、dataset/labels/{train,val}、
                          runs/、train.py、predict.py、data.yaml
  【提交规范建议结构】README.md、requirements.txt、results/、report.md
另外做几项内容检查：data.yaml 的字段与类别名、图片/标签文件名配对情况、标签类别分布。
输出可直接截图作为 Level 1 的“目录结构”验证材料。
"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

LEVEL1 = [
    ('dataset/images/train', 'dir'),
    ('dataset/images/val', 'dir'),
    ('dataset/labels/train', 'dir'),
    ('dataset/labels/val', 'dir'),
    ('runs', 'dir'),
    ('train.py', 'file'),
    ('predict.py', 'file'),
    ('data.yaml', 'file'),
]
SUBMIT = [
    ('README.md', 'file'),
    ('requirements.txt', 'file'),
    ('report.md', 'file'),
    ('results', 'dir'),
]


def check_items(title, items):
    print('【%s】' % title)
    ok_all = True
    for rel, kind in items:
        p = PROJ / rel
        exists = p.is_dir() if kind == 'dir' else p.is_file()
        n = ''
        if exists and p.is_dir():
            n = ' (%d files)' % sum(1 for x in p.rglob('*') if x.is_file())
        elif exists and p.is_file():
            n = ' (%.1f KB)' % (p.stat().st_size / 1024)
        print('  [%s] %-26s%s' % ('PASS' if exists else 'FAIL', rel, n))
        ok_all = ok_all and exists
    print()
    return ok_all


def check_data_yaml():
    print('【data.yaml 内容检查】')
    f = PROJ / 'data.yaml'
    if not f.exists():
        print('  [FAIL] 文件不存在')
        return False
    text = f.read_text(encoding='utf-8')
    ok = True
    for key in ('path:', 'train:', 'val:', 'names:'):
        has = key in text
        print('  [%s] 含字段 %s' % ('PASS' if has else 'FAIL', key))
        ok = ok and has
    for cid, name in ((0, 'obstacle'), (1, 'cola'), (2, 'football')):
        has = ('%d: %s' % (cid, name)) in text
        print('  [%s] 类别 %d = %s' % ('PASS' if has else 'FAIL', cid, name))
        ok = ok and has
    print()
    return ok


def check_pairs():
    print('【图片 / 标签 配对检查】')
    ok = True
    for split in ('train', 'val'):
        img_dir = PROJ / 'dataset' / 'images' / split
        lbl_dir = PROJ / 'dataset' / 'labels' / split
        imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in EXTS} if img_dir.exists() else set()
        lbls = {p.stem for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'} if lbl_dir.exists() else set()
        extra = lbls - imgs
        labeled = len(imgs & lbls)
        print('  %-5s images=%d  labels=%d  已配对=%d  无标签=%d  多余标签=%d'
              % (split, len(imgs), len(lbls), labeled, len(imgs) - labeled, len(extra)))
        if extra:
            print('        [FAIL] 存在找不到对应图片的标签（放错 split？）：%s'
                  % ', '.join(sorted(extra)[:5]))
            ok = False
    print()
    print('  说明：无标签的图片会在训练时被当作“背景图”，补齐前不要正式训练。')
    print()
    return ok


def main():
    print('项目目录: %s' % PROJ)
    print('=' * 62)
    a = check_items('Level 1 推荐目录结构', LEVEL1)
    b = check_items('提交规范建议结构', SUBMIT)
    c = check_data_yaml()
    d = check_pairs()
    print('=' * 62)
    if a and b and c and d:
        print('RESULT: PASS —— 目录结构完全符合考核要求，可截图作为 Level 1 验证材料')
        return 0
    print('RESULT: FAIL —— 上面标 FAIL 的项需要补齐')
    return 1


if __name__ == '__main__':
    sys.exit(main())
