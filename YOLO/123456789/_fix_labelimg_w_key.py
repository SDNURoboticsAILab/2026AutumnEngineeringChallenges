# -*- coding: utf-8 -*-
"""修复 labelImg 按 W 闪退（两个问题一起修）。

根因（已用脚本复现，退出码 0xC0000409 = PyQt5 遇到未捕获异常直接终止进程）：
  1) 两个动作都绑定了同一个快捷键 W：
        create_mode -> self.set_create_mode   （本意是"高级模式"的多边形绘制）
        create      -> self.create_shape      （真正要用的矩形框）
     Qt 遇到重复快捷键时只触发其中一个，实测触发了 set_create_mode；
  2) set_create_mode() 第一行是 `assert self.advanced()`，默认的 beginner 模式下必然
     AssertionError；异常发生在 Qt 槽函数里，PyQt5 会直接 qFatal 终止 → 表现为"按 W 就闪退"。

修复：
  A. 把 create_mode 的快捷键从 'w' 改成 'Ctrl+Shift+W'，让 W 唯一对应 create_shape；
  B. 把 set_create_mode / set_edit_mode 里的 assert 改成优雅返回，避免任何路径下炸掉进程。

用法（需要更宽权限，因为要改虚拟环境里的 site-packages）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe _fix_labelimg_w_key.py
"""
import re
import shutil
import sys
from pathlib import Path

TARGETS = [
    Path(r'C:\Users\dell\label-env\Lib\site-packages\labelImg\labelImg.py'),   # 专用标注环境 py3.11
    Path(r'C:\Users\dell\yolo-env\Lib\site-packages\labelImg\labelImg.py'),    # 训练环境 py3.14
]

OLD_ASSERT_BLOCK = (
    "    def set_create_mode(self):\n"
    "        assert self.advanced()\n"
    "        self.toggle_draw_mode(False)\n"
    "\n"
    "    def set_edit_mode(self):\n"
    "        assert self.advanced()\n"
    "        self.toggle_draw_mode(True)"
)
NEW_ASSERT_BLOCK = (
    "    def set_create_mode(self):\n"
    "        if not self.advanced():\n"
    "            return\n"
    "        self.toggle_draw_mode(False)\n"
    "\n"
    "    def set_edit_mode(self):\n"
    "        if not self.advanced():\n"
    "            return\n"
    "        self.toggle_draw_mode(True)"
)


def patch(path):
    if not path.exists():
        print('[SKIP] 不存在: %s' % path)
        return
    bak = path.with_suffix('.py.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        print('  备份 -> %s' % bak)

    text = path.read_text(encoding='utf-8')

    # A. 解除 W 的重复绑定
    if "'Ctrl+Shift+W'" in text:
        print('  [SKIP] 快捷键已修')
    else:
        text, n = re.subn(r"(self\.set_create_mode,\s*\n\s*)'w'",
                          r"\1'Ctrl+Shift+W'", text)
        print('  [FIX ] create_mode 快捷键 w -> Ctrl+Shift+W（命中 %d 处）' % n)

    # B. assert -> 优雅返回
    if NEW_ASSERT_BLOCK in text:
        print('  [SKIP] assert 已修')
    elif OLD_ASSERT_BLOCK in text:
        text = text.replace(OLD_ASSERT_BLOCK, NEW_ASSERT_BLOCK)
        print('  [FIX ] set_create_mode / set_edit_mode 的 assert 改为优雅返回')
    else:
        print('  [MISS] 未找到 assert 代码块（可能版本不同）')

    path.write_text(text, encoding='utf-8')
    print('  写入完成: %s' % path)


def main():
    for p in TARGETS:
        print('=== %s ===' % p)
        patch(p)
    print('\n完成。请重新启动 labelImg 试按 W。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
