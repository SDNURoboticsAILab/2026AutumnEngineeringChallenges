# -*- coding: utf-8 -*-
"""打印项目目录结构（用于 Level 1 要求的“目录结构截图”）。

用法（在 yolo_project 下）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe show_structure.py
结果同时打印到屏幕并写入 structure.txt。
"""
from pathlib import Path

PROJ = Path(__file__).resolve().parent
SKIP = {'.venv', '__pycache__', '.git', '.idea'}


def walk(path, prefix='', depth=0):
    lines = []
    entries = [p for p in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
               if p.name not in SKIP]
    for i, p in enumerate(entries):
        last = i == len(entries) - 1
        branch = '+-- ' if last else '|-- '
        if p.is_dir():
            n = sum(1 for _ in p.rglob('*') if _.is_file())
            lines.append('%s%s%s/   [%d files]' % (prefix, branch, p.name, n))
            # 前三层全部展开（dataset/images/train 这种结构要能看到），再深就只显示数量
            if depth < 2:
                lines += walk(p, prefix + ('    ' if last else '|   '), depth + 1)
        else:
            size = p.stat().st_size
            lines.append('%s%s%-28s %8s' % (prefix, branch, p.name, human(size)))
    return lines


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return '%.0f %s' % (n, unit) if unit == 'B' else '%.1f %s' % (n, unit)
        n /= 1024
    return '%.1f TB' % n


def main():
    lines = ['%s/' % PROJ.name] + walk(PROJ)
    text = '\n'.join(lines)
    print(text)
    (PROJ / 'structure.txt').write_text(text + '\n', encoding='utf-8')
    print('\n（已写入 structure.txt，可直接截图用于报告）')


if __name__ == '__main__':
    main()
