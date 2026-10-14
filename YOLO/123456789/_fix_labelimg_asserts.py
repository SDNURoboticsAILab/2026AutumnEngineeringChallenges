# -*- coding: utf-8 -*-
"""把 labelImg 里剩下的 assert 全部改成优雅处理（避免又一个"闪退"）。

为什么：PyQt5 在 Qt 槽函数里遇到未捕获异常会直接终止进程（闪退、无报错）。
labelImg 里剩下的三个 assert 都在"按 W 画框 / 重绘画布 / 另存为"这些热路径上：

  1) create_shape():        assert self.beginner()
     —— 高级模式（Ctrl+Shift+A 打开过）下 beginner() 为 False，按 W 必然 AssertionError → 闪退
  2) paint_canvas():        assert not self.image.isNull()
     —— 没有任何图片时触发重绘 → 闪退
  3) save_file_as():        assert not self.image.isNull()
     —— 没有图片时另存为 → 闪退

用法（需要更宽权限，因为要改虚拟环境里的 site-packages）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe _fix_labelimg_asserts.py
"""
import shutil
import sys
from pathlib import Path

TARGETS = [
    Path(r'C:\Users\dell\label-env\Lib\site-packages\labelImg\labelImg.py'),
    Path(r'C:\Users\dell\yolo-env\Lib\site-packages\labelImg\labelImg.py'),
]

PATCHES = [
    # 1) create_shape：去掉 assert，两边模式都能正常进入画框
    ("    def create_shape(self):\n        assert self.beginner()\n        self.canvas.set_editing(False)",
     "    def create_shape(self):\n        # 原代码是 assert self.beginner()，高级模式下会抛 AssertionError 并让进程直接退出\n        self.canvas.set_editing(False)"),
    # 2) paint_canvas：空图时直接返回
    ('    def paint_canvas(self):\n        assert not self.image.isNull(), "cannot paint null image"\n',
     '    def paint_canvas(self):\n        if self.image.isNull():\n            return\n'),
    # 3) save_file_as：空图时直接返回
    ('    def save_file_as(self, _value=False):\n        assert not self.image.isNull(), "cannot save empty image"\n',
     '    def save_file_as(self, _value=False):\n        if self.image.isNull():\n            return\n'),
]


def patch(path):
    if not path.exists():
        print('[SKIP] 不存在: %s' % path)
        return
    bak = path.with_suffix('.py.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        print('  备份 -> %s' % bak)
    text = path.read_text(encoding='utf-8')
    changed = 0
    for old, new in PATCHES:
        if new in text:
            print('  [SKIP] 已修复: %s' % new.strip().splitlines()[-1][:50])
            continue
        if old in text:
            text = text.replace(old, new)
            changed += 1
            print('  [FIX ] %s' % new.strip().splitlines()[-1][:60])
        else:
            print('  [MISS] 未找到: %s' % old.strip().splitlines()[-1][:50])
    path.write_text(text, encoding='utf-8')
    left = text.count('assert ')
    print('  写入完成；文件里剩余 assert 语句: %d 处' % left)


def main():
    for p in TARGETS:
        print('=== %s ===' % p)
        patch(p)
    print('\n完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
