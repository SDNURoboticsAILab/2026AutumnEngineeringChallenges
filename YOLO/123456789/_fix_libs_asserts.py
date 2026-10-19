# -*- coding: utf-8 -*-
"""清掉 libs 里剩下的断言（都是会把进程直接搞死的隐患）。

canvas.py:
    set_last_label():   assert text        -> 空标签（对话框直接回车）时会闪退
    undo_last_line():   assert self.shapes -> Ctrl+Z 且没有框时闪退
    reset_all_lines():  assert self.shapes -> 取消创建且没有框时闪退
shape.py:
    draw_vertex():      assert False, "unsupported vertex shape" -> 画不支持的顶点类型时闪退

用法（需要更宽权限）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe _fix_libs_asserts.py
"""
import shutil
import sys
from pathlib import Path

ENVS = [
    Path(r'C:\Users\dell\label-env\Lib\site-packages'),
    Path(r'C:\Users\dell\yolo-env\Lib\site-packages'),
]

CANVAS_PATCHES = [
    ("    def set_last_label(self, text, line_color=None, fill_color=None):\n        assert text\n",
     "    def set_last_label(self, text, line_color=None, fill_color=None):\n"
     "        if not text:          # 空标签（对话框里直接回车）时不再让进程退出\n            return None\n"),
    ("    def undo_last_line(self):\n        assert self.shapes\n",
     "    def undo_last_line(self):\n        if not self.shapes:\n            return\n"),
    ("    def reset_all_lines(self):\n        assert self.shapes\n",
     "    def reset_all_lines(self):\n"
     "        if not self.shapes:\n"
     "            self.current = None\n"
     "            self.drawingPolygon.emit(False)\n"
     "            self.update()\n"
     "            return\n"),
]

SHAPE_PATCHES = [
    ('            assert False, "unsupported vertex shape"',
     '            return  # 不支持的顶点类型：跳过绘制而不是结束进程'),
]


def patch_file(path, patches):
    if not path.exists():
        print('  [SKIP] 不存在: %s' % path)
        return
    bak = path.with_suffix('.py.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
    text = path.read_text(encoding='utf-8')
    for old, new in patches:
        if new in text:
            print('  [SKIP] 已修复: %s' % new.strip().splitlines()[0][:56])
            continue
        if old in text:
            text = text.replace(old, new)
            print('  [FIX ] %s' % new.strip().splitlines()[0][:56])
        else:
            print('  [MISS] 未找到: %s' % old.strip().splitlines()[0][:56])
    path.write_text(text, encoding='utf-8')
    print('  剩余 assert: %d 处' % text.count('assert '))


def main():
    for env in ENVS:
        print('=== %s ===' % env)
        patch_file(env / 'libs' / 'canvas.py', CANVAS_PATCHES)
        patch_file(env / 'libs' / 'shape.py', SHAPE_PATCHES)
    print('\n完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
