"""YOLOv8 训练脚本 —— obstacle / cola / football 三分类目标检测。

用法：
    python train.py                      # 使用默认参数训练
    python train.py --name exp2 --epochs 120
    python train.py --benchmark          # 只跑 2 轮，用来估算单轮耗时

说明：
    本机显卡为 AMD Radeon RX 7700 XT，没有 CUDA 支持，因此训练在 CPU 上进行。
    相应地把模型换成最小的 yolov8n、batch 调小、并开启多进程数据加载，
    在可接受的时间内拿到可用精度。
"""

from __future__ import annotations

import argparse
import os
import time


def enable_serial_label_scan() -> None:
    """在受限环境里把 ultralytics 的并行标签扫描改成串行。

    ultralytics 扫描数据集标签时会用 `multiprocessing.pool.ThreadPool` 并行处理；
    这个线程池在 Windows 上底层要创建**命名管道**。普通桌面环境没问题，
    但在禁止创建命名管道的受限环境（部分沙箱 / 容器）里会直接抛：

        PermissionError: [WinError 5]  ... _winapi.CreateFile ...

    这里把 ThreadPool 换成一个接口兼容的串行实现：行为完全一致，
    只是扫描 949 张标签多花几秒。**正常环境不需要加 --serial-scan。**
    注意必须在 import ultralytics 之前打补丁才生效。
    """
    import multiprocessing.pool as mpp

    class _SerialPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap(self, func, iterable, chunksize=None):
            return map(func, iterable)

        def imap_unordered(self, func, iterable, chunksize=None):
            return map(func, iterable)

        def map(self, func, iterable, chunksize=None):
            return list(map(func, iterable))

        def close(self):
            pass

        def join(self):
            pass

        def terminate(self):
            pass

    mpp.ThreadPool = _SerialPool
    print("已启用串行标签扫描（受限环境兼容模式）")


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 YOLOv8 目标检测模型")
    ap.add_argument("--weights", default="yolov8n.pt", help="预训练权重（迁移学习起点）")
    ap.add_argument("--data", default="data.yaml", help="数据集配置文件")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu 或 0（本机无 CUDA，只能用 cpu）")
    ap.add_argument("--workers", type=int, default=8, help="数据加载进程数")
    ap.add_argument("--name", default="exp1", help="本次实验的名字，结果存到 runs/<name>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=25, help="多少轮没提升就早停")
    ap.add_argument("--benchmark", action="store_true", help="只跑 2 轮，估算单轮耗时")
    ap.add_argument("--serial-scan", action="store_true",
                    help="受限环境兼容：串行扫描标签（普通环境不要加）")
    args = ap.parse_args()

    if args.serial_scan:
        enable_serial_label_scan()

    epochs = 2 if args.benchmark else args.epochs

    # CPU 训练时把线程数用满（16 个逻辑核）
    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 8)
    except Exception:  # noqa: BLE001
        pass

    from ultralytics import YOLO

    # 结果目录用脚本所在位置的绝对路径。
    # 不能只写 "runs"：ultralytics 会把相对的 project 拼到它自己的 runs_dir 设置下，
    # 结果会跑到 runs/detect/runs/... 这种嵌套目录里，后续 predict.py 就找不到了。
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

    model = YOLO(args.weights)
    t0 = time.time()
    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=project_dir,
        name=args.name,
        exist_ok=True,
        plots=True,          # 生成 results.png / confusion_matrix.png / PR 曲线，报告要用
        val=True,
        amp=False,           # CPU 上关闭混合精度
        cache=False,         # 不缓存到内存，避免占用过多内存
        verbose=True,
    )
    dt = time.time() - t0
    print(f"\n训练结束，用时 {dt/60:.1f} 分钟（{epochs} 轮，平均 {dt/epochs:.1f} 秒/轮）")
    print(f"权重文件：runs/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
