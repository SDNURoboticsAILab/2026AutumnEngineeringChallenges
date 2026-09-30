# -*- coding: utf-8 -*-
"""
run_train_shim.py —— 训练用兼容性启动器（仅用于受限环境/无命名管道环境）
================================================================================
【为什么需要这个文件】
train.py 本身是标准代码，直接 `python train.py` 在正常 Windows 上是没问题的。
但在受限沙箱环境里会失败，原因是：

    ultralytics/data/dataset.py 第 132 行使用
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(...)
    来并行建立标签缓存。CPython 的 multiprocessing.pool.ThreadPool 内部会创建
    SimpleQueue，而 SimpleQueue 需要打开【命名管道】。受限环境禁止创建命名管道，
    于是抛出：
        PermissionError: [WinError 5] 拒绝访问。
                      ... multiprocessing/connection.py ... _winapi.CreateFile(...)

【本启动器做什么】
在导入 ultralytics 之前，把 multiprocessing.pool.ThreadPool 替换成一个
"就地顺序执行"的实现（same API，无进程/无队列/无管道）。
因为被并行化的只是"读标签文件、读图片"这类互不依赖的小任务，
顺序执行的结果与并行完全一致，只是慢一点。

【它与 train.py 的关系】
不修改 train.py 一行代码。用法：
    python run_train_shim.py --epochs 30 --imgsz 320 --batch 2 --device cpu
参数与 train.py 完全一致，会被原样转发。
"""

from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1) 安装 ThreadPool 兼容垫片（必须在导入 ultralytics 之前完成）
# ---------------------------------------------------------------------------
class _SyncPool:
    """ThreadPool 的直接替身：imap 直接就地顺序执行，不建进程/队列/管道。

    只实现 ultralytics 实际用到的接口：上下文管理协议、imap、close、join。
    """

    def __init__(self, *args, **kwargs):
        pass                                  # 忽略 processes 参数

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False                          # 不吞异常

    def imap(self, func, iterable, chunksize=None):
        # 惰性逐条执行：调用方 for 循环拿到一项时，这一项已经算好了
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
    import multiprocessing.pool as _mp_pool
    _mp_pool.ThreadPool = _SyncPool
    # 有些模块用 from multiprocessing.pool import ThreadPool，
    # 走的是属性查找，所以替换模块属性即可全局生效。
    print("[shim] ThreadPool 已替换为顺序执行版本（绕过命名管道限制）")
except Exception as e:                       # pragma: no cover
    print(f"[shim] 安装 ThreadPool 垫片失败: {e!r}")

try:
    import multiprocessing
    multiprocessing.pool.ThreadPool = _SyncPool
except Exception:
    pass


# ---------------------------------------------------------------------------
# 2) 转发命令行参数给 train.py 的 main()
# ---------------------------------------------------------------------------
# 本文件位于 results/code/ 下，因此需要把两个目录都加入模块搜索路径：
#   - 项目根（你的学号/）：train.py、data.yaml 在这里
#   - 本文件所在目录：仅用于兜底
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # 上溯两级到项目根
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as train_mod                    # noqa: E402  导入项目根下的 train.py

if __name__ == "__main__":
    train_mod.main()
