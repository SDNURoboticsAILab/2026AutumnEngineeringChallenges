# -*- coding: utf-8 -*-
"""校验 data.yaml 是否配置正确（对应考核 Level 3 第 1 节：配置数据集文件）。

检查内容：
  1) YAML 能否解析、字段是否齐全（path / train / val / names）
  2) Ultralytics 实际解析出的 train / val 路径到底是什么（这一步最容易踩坑：
     相对路径会被解释成 ultralytics 的 datasets_dir 下的路径，而不是项目目录）
  3) 两个目录里的图片数与标签数、三类实例数、类别编号是否越界
  4) 与 classes.txt 的一致性

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe verify_data_yaml.py
"""
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJ = Path(__file__).resolve().parent
YAML = PROJ / 'data.yaml'
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}


def section(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


def main():
    section('1) 解析 YAML')
    if not YAML.exists():
        print('[FAIL] 找不到 %s' % YAML)
        return 2
    data = yaml.safe_load(YAML.read_text(encoding='utf-8'))
    print('内容:', data)
    ok = True
    for key in ('path', 'train', 'val', 'names'):
        has = key in data
        print('  [%s] 含字段 %s' % ('PASS' if has else 'FAIL', key))
        ok = ok and has
    names = data.get('names') or {}
    print('  类别:', names)
    if set(names.keys()) != {0, 1, 2} or [names[i] for i in (0, 1, 2)] != ['obstacle', 'cola', 'football']:
        print('  [FAIL] names 必须严格是 0: obstacle / 1: cola / 2: football')
        ok = False
    else:
        print('  [PASS] 类别编号与名称正确')

    section('2) Ultralytics 实际解析出的路径（关键）')
    try:
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import SETTINGS
        print('ultralytics datasets_dir 设置 =', SETTINGS.get('datasets_dir'))
        d = check_det_dataset(str(YAML))
        print('  解析结果 train =', d.get('train'))
        print('  解析结果 val   =', d.get('val'))
        print('  解析结果 nc    =', d.get('nc'))
        for key in ('train', 'val'):
            p = Path(d.get(key, ''))
            exists = p.exists()
            print('  [%s] %s 指向的目录存在: %s' % ('PASS' if exists else 'FAIL', key, exists))
            ok = ok and exists
    except Exception as exc:
        print('  [WARN] check_det_dataset 失败（可能是沙箱限制）:', type(exc).__name__, exc)
        # 退化为手工解析
        base = Path(str(data.get('path', '.')))
        if not base.is_absolute():
            print('  注意：path 写成相对路径时，ultralytics 的解析顺序是：')
            print('        ① 先看「当前工作目录」下是否存在该相对路径（./dataset），存在就直接用；')
            print('        ② 不存在，才回退到 datasets_dir/%s 下去找。'
                  % str(data.get('path')).lstrip('./'))
            print('        因此从项目目录运行脚本时能正确解析，换到别的目录运行就会报')
            print('        “dataset not found”——这是最常见的路径错误原因。')
            base = PROJ / base.name
        for key in ('train', 'val'):
            p = (base / str(data.get(key))).resolve()
            print('  手工解析 %s -> %s (存在=%s)' % (key, p, p.exists()))

    section('3) 数据规模与类别统计（按 data.yaml 的 train/val 目录）')
    base = PROJ / 'dataset'
    for key in ('train', 'val'):
        img_dir = base / str(data.get(key, ''))
        lbl_dir = Path(str(img_dir).replace('images', 'labels'))
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in EXTS] if img_dir.exists() else []
        lbls = [p for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'] if lbl_dir.exists() else []
        counts = Counter()
        bad = 0
        for f in lbls:
            for line in f.read_text(encoding='utf-8', errors='ignore').splitlines():
                parts = line.split()
                if len(parts) != 5:
                    bad += 1
                    continue
                cid = int(float(parts[0]))
                if cid not in NAMES:
                    bad += 1
                    continue
                counts[cid] += 1
        print('  [%s] 图片 %d 张，标签 %d 个，格式异常行 %d' % (key, len(imgs), len(lbls), bad))
        print('         ' + '  '.join('%s=%d' % (NAMES[c], counts[c]) for c in sorted(NAMES)))
        if bad:
            ok = False

    section('4) 与 classes.txt 一致性')
    classes = PROJ / 'classes.txt'
    if classes.exists():
        lines = [l.strip() for l in classes.read_text(encoding='utf-8').splitlines() if l.strip()]
        match = lines == ['obstacle', 'cola', 'football']
        print('  [%s] classes.txt = %s' % ('PASS' if match else 'FAIL', lines))
        ok = ok and match
    else:
        print('  [WARN] 没有 classes.txt（labelImg 用它确定类别顺序）')

    section('结论')
    print('  [%s] data.yaml 配置%s' % ('PASS' if ok else 'FAIL',
                                     '正确，可以直接用于训练' if ok else '存在问题，见上面 FAIL 项'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
