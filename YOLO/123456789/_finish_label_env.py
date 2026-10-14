# -*- coding: utf-8 -*-
"""收尾：在 Python 3.11 专用标注环境里装 labelImg 并打上 int() 补丁。

背景（已实测）：PyQt5 的绑定在 5.15.6 / 5.15.9 + sip 12.11/12.12/12.13 下
同样拒收 float，所以“免补丁”的组合不存在；补丁是必需的一步。
本脚本让专用环境也可以直接使用。

执行（需要更宽权限）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe _finish_label_env.py
"""
import pathlib
import shutil
import subprocess
import sys

VENV = pathlib.Path(r'C:\Users\dell\label-env')
LPY = VENV / 'Scripts' / 'python.exe'
LIMG = VENV / 'Scripts' / 'labelImg.exe'
SRC = VENV / 'Lib' / 'site-packages' / 'labelImg' / 'labelImg.py'

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


def run(cmd, capture=True):
    cmd = [str(c) for c in cmd]
    print('>> ' + ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = ((p.stdout or '') + (p.stderr or '')).strip()
    print('   exit = %d' % p.returncode)
    for line in out.splitlines()[-8:]:
        print('   | ' + line)
    return p.returncode, out


def main():
    print('=== [1/3] 在专用环境安装 labelImg ===', flush=True)
    run([LPY, '-m', 'pip', 'install', '--no-cache-dir', 'labelImg'])
    if not SRC.exists():
        print('[FAIL] 找不到 %s' % SRC)
        return 2

    print('\n=== [2/3] 给 labelImg.py 打 int() 补丁 ===', flush=True)
    bak = SRC.with_suffix('.py.bak')
    if not bak.exists():
        shutil.copy2(SRC, bak)
        print('已备份 -> %s' % bak)
    text = SRC.read_text(encoding='utf-8')
    for old, new in PATCHES:
        if new in text:
            print('[SKIP] 已修复: %s' % new[:56])
        elif old in text:
            text = text.replace(old, new)
            print('[FIX ] %s' % new[:56])
        else:
            print('[MISS] 未找到: %s' % old[:56])
    SRC.write_text(text, encoding='utf-8')

    print('\n=== [3/3] 校验 ===', flush=True)
    run([LPY, '-m', 'pip', 'list'])
    run([LIMG, '--help'])
    print('\n专用标注环境就绪: %s' % LIMG)
    return 0


if __name__ == '__main__':
    sys.exit(main())
