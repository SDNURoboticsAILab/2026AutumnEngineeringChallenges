# -*- coding: utf-8 -*-
"""修复“按 W 拖动就闪退”的真正原因：QPainter 的 drawRect / drawLine / drawText 收到 float 参数。

实测（本机 PyQt5 5.15.11 + sip 12.19）：
    painter.drawRect(1.5, 2.5, 3.5, 4.5)   -> TypeError: arguments did not match any overloaded call
    painter.drawLine(1.5, 0, 1.5, 100)     -> TypeError
    painter.drawText(1.5, 2.5, "hi")       -> TypeError

而 labelImg 在“画框”时正好这么调用：
    libs/canvas.py:526   p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)   <- 橡皮筋选边框
    libs/canvas.py:530-531  p.drawLine(self.prev_point.x(), 0, ...)                        <- 十字准线
    libs/shape.py:131    painter.drawText(min_x, min_y, self.label)                        <- 框上的类别文字

这些代码在重绘（paintEvent）里执行 —— 一旦鼠标在画框状态下移动，画布重绘 ->
drawRect 收到 float -> TypeError -> 异常发生在 Qt 事件处理里 -> PyQt5 直接终止进程（闪退）。

修复：把这些坐标转成 int（橡皮筋/准线用整数坐标完全看不出差别）。
另外顺手把 canvas.py 里剩下的两个 assert 改成优雅返回，彻底消除“断言失败=进程退出”。

用法（需要更宽权限，因为要改虚拟环境里的 site-packages）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe _fix_labelimg_painter.py
"""
import shutil
import sys
from pathlib import Path

ENVS = [
    Path(r'C:\Users\dell\label-env\Lib\site-packages'),
    Path(r'C:\Users\dell\yolo-env\Lib\site-packages'),
]

CANVAS_PATCHES = [
    ("            p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
     "            p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))"),
    ("            p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())\n"
     "            p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
     "            p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), self.pixmap.height())\n"
     "            p.drawLine(0, int(self.prev_point.y()), self.pixmap.width(), int(self.prev_point.y()))"),
    ("    def end_move(self, copy=False):\n        assert self.selected_shape and self.selected_shape_copy",
     "    def end_move(self, copy=False):\n        if not (self.selected_shape and self.selected_shape_copy):\n            return"),
    ("    def finalise(self):\n        assert self.current",
     "    def finalise(self):\n        if not self.current:\n            return"),
]

SHAPE_PATCHES = [
    ("                    painter.drawText(min_x, min_y, self.label)",
     "                    painter.drawText(int(min_x), int(min_y), self.label)"),
]


def patch_file(path, patches):
    if not path.exists():
        print('  [SKIP] 不存在: %s' % path)
        return
    bak = path.with_suffix('.py.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        print('  备份 -> %s' % bak.name)
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


def main():
    for env in ENVS:
        print('=== %s ===' % env)
        patch_file(env / 'libs' / 'canvas.py', CANVAS_PATCHES)
        patch_file(env / 'libs' / 'shape.py', SHAPE_PATCHES)
    print('\n完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
