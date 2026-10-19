# -*- coding: utf-8 -*-
"""让 labelImg 能加载已有 YOLO 标签 + 让自检脚本忽略 classes.txt。

背景（实测踩到的坑）：
    labelImg 的 YoloReader 在读取已有 YOLO txt 时，会去**标签所在目录**找
    classes.txt（libs/yolo_io.py: os.path.dirname(txt_path)/classes.txt），
    找不到就直接 FileNotFoundError 崩溃。所以必须在标签目录里也放一份 classes.txt。

    但这又会让"把标签目录里所有 *.txt 都当标签"的自检脚本把它当成多余/孤儿标签，
    因此本脚本同时给这些脚本加上排除规则。

用法（在 yolo_project 下）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe _fix_classes_txt.py
"""
import pathlib
import shutil

PROJ = pathlib.Path(__file__).resolve().parent
CLASSES = PROJ / 'classes.txt'
LABEL_DIRS = [PROJ / 'dataset' / 'labels' / 'train', PROJ / 'dataset' / 'labels' / 'val']

# (文件, 原字符串, 新字符串)
PATCHES = [
    ('check_labels.py',
     "for p in lbl_dir.glob('*.txt')}",
     "for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'}"),
    ('check_rawdata.py',
     "for p in lbl_dir.glob('*.txt')}",
     "for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'}"),
    ('verify_structure.py',
     "for p in lbl_dir.glob('*.txt')}",
     "for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'}"),
    ('fix_label_placement.py',
     "for p in lbl_dir.glob('*.txt')}",
     "for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'}"),
    ('fix_label_placement.py',
     "sorted(lbl_dir.glob('*.txt'))",
     "sorted(p for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt')"),
    ('check_transfer.py',
     "list(lbl_dir.glob('*.txt'))",
     "[p for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt']"),
]


def main():
    print('=== [1/2] 把 classes.txt 复制到标签目录（labelImg 需要）===')
    if not CLASSES.exists():
        raise SystemExit('找不到 %s' % CLASSES)
    for d in LABEL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        dst = d / 'classes.txt'
        shutil.copy2(CLASSES, dst)
        print('  写入 %s（%d 行）' % (dst, len(dst.read_text(encoding='utf-8').splitlines())))

    print('\n=== [2/2] 让自检脚本忽略 classes.txt ===')
    for fname, old, new in PATCHES:
        path = PROJ / fname
        if not path.exists():
            print('  [SKIP] %s 不存在' % fname)
            continue
        text = path.read_text(encoding='utf-8')
        if new in text:
            print('  [SKIP] %s 已处理' % fname)
            continue
        if old not in text:
            print('  [MISS] %s 未找到: %s' % (fname, old[:50]))
            continue
        path.write_text(text.replace(old, new), encoding='utf-8')
        print('  [FIX ] %s' % fname)

    print('\n完成。接下来请依次运行：check_labels.py / check_transfer.py / audit_labels.py / batch_progress.py')


if __name__ == '__main__':
    main()
