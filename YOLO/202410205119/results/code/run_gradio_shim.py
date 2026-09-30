#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_gradio_shim.py —— gradio_app.py 的兼容性启动器（仅在受限环境下需要）
================================================================================
【为什么需要】
gradio_app.py 本身是标准代码。但在禁止创建命名管道的受限环境里，
ultralytics 建立标签缓存时会用到 multiprocessing.pool.ThreadPool，
而 ThreadPool 内部的 SimpleQueue 需要打开命名管道，于是报：
    PermissionError: [WinError 5] 拒绝访问。

本启动器在导入 ultralytics 之前，把 ThreadPool 替换成"就地顺序执行"的实现，
从而绕过命名管道限制。被并行化的只是读图/读标签这类互不依赖的小任务，
顺序执行的结果与并行完全一致，只是慢一点。

【与 gradio_app.py 的关系】
不修改 gradio_app.py 一行代码。用法与它完全相同：
    python run_gradio_shim.py --port 7860
"""

from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1) 安装 ThreadPool 兼容垫片（必须在导入 ultralytics 之前完成）
# ---------------------------------------------------------------------------
class _SyncPool:
    """ThreadPool 的替身：imap 就地顺序执行，不创建进程/队列/管道。"""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, func, iterable, chunksize=None):
        for item in iterable:
            yield func(item)

    def map(self, func, iterable, chunksize=None):
        return [func(i) for i in iterable]

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


try:
    import multiprocessing
    import multiprocessing.pool as _mp_pool
    _mp_pool.ThreadPool = _SyncPool
    multiprocessing.pool.ThreadPool = _SyncPool
    print("[shim] ThreadPool 已替换为顺序执行版本（绕过命名管道限制）")
except Exception as e:                                   # pragma: no cover
    print(f"[shim] 安装 ThreadPool 垫片失败: {e!r}")


# ---------------------------------------------------------------------------
# 2) 转发参数给 gradio_app.main()
# ---------------------------------------------------------------------------
# 本文件位于 results/code/ 下，gradio_app.py 也在同一目录，
# 同时把项目根加入搜索路径，保证 gradio_app 能正确解析 data.yaml / runs。
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # 上溯两级到项目根
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio_app                                        # noqa: E402

if __name__ == "__main__":
    gradio_app.main()
