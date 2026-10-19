# -*- coding: utf-8 -*-
"""修复 labelImg 1.8.6 在 PyQt5 5.15.11 下的 float/int 崩溃。

问题：labelImg 里几处把浮点数直接传给 QScrollBar/QSpinBox 的 setValue()，
新版 sip 不再自动把 float 转成 int，于是抛
    TypeError: setValue(self, a0: int): argument 1 has unexpected type 'float'
（鼠标滚轮一滚就崩，窗口刚打开也可能因为滚动事件直接退出。）

修复：给这几处补上 int() 转换（与上游后续版本的改法一致）：
    1. scroll_request() 里 bar.setValue(bar.value() + bar.singleStep() * units)
    2. set_zoom() 里 self.zoom_widget.setValue(value)
    3. zoom_request() 里 h_bar.setValue(new_h_bar_value) / v_bar.setValue(new_v_bar_value)

用法（需要写 site-packages，请用管理员/普通权限的终端运行）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe _patch_labelimg.py
"""
import shutil
from pathlib import Path

TARGET = Path(r'C:\Users\dell\yolo-env\Lib\site-packages\labelImg\labelImg.py')

PATCHES = [
    ('bar.setValue(bar.value() + bar.singleStep() * units)',
     'bar.setValue(int(bar.value() + bar.singleStep() * units))'),
    ('self.zoom_widget.setValue(value)',
     'self.zoom_widget.setValue(int(value))'),
    ('h_bar.setValue(new_h_bar_value)',
     'h_bar.setValue(int(new_h_bar_value))'),
    ('v_bar.setValue(new_v_bar_value)',
     'v_bar.setValue(int(new_v_bar_value))'),
]


def main():
    if not TARGET.exists():
        raise SystemExit('找不到文件: %s' % TARGET)

    backup = TARGET.with_suffix('.py.bak')
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print('已备份原文件 -> %s' % backup)
    else:
        print('备份已存在 -> %s' % backup)

    text = TARGET.read_text(encoding='utf-8')
    for old, new in PATCHES:
        if new in text:
            print('[SKIP] 已修复: %s' % new[:60])
            continue
        if old in text:
            text = text.replace(old, new)
            print('[FIX ] %s' % new[:60])
        else:
            print('[MISS] 未找到待替换代码: %s' % old[:60])

    TARGET.write_text(text, encoding='utf-8')
    print('写入完成: %s' % TARGET)
    print('如果之后想还原：copy "%s" "%s"' % (backup, TARGET))


if __name__ == '__main__':
    main()
