"""阶段 3：训练 YOLO11 模型（Level 3）

本机是 Intel Arc 核显、无 NVIDIA/CUDA，所以固定用 CPU 训练：
  device=cpu、workers=0（受限环境多进程会 WinError 5）、迁移学习（yolo11n.pt 预训练权重）

用法：
    # 先短跑 3 个 epoch 确认流程没问题（约 10~20 分钟）
    D:\\YOLO\\.venv\\Scripts\\python.exe train.py --epochs 3 --name quick_test

    # 正式训练（建议夜间挂机，约 4~8 小时）
    D:\\YOLO\\.venv\\Scripts\\python.exe train.py --epochs 60 --name yolo11n_cpu

    # 内存充足（>= 8GB 空闲）时开缓存，能明显加速
    D:\\YOLO\\.venv\\Scripts\\python.exe train.py --epochs 60 --cache

训练产物：runs/train/<name>/
    results.csv / results.png       Loss、Precision、Recall、mAP 曲线（报告要用的截图）
    confusion_matrix.png            混淆矩阵
    weights/best.pt  weights/last.pt  权重
    val_batch*.jpg                  验证集预测可视化
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def keep_awake() -> bool:
    """训练期间禁止系统睡眠/休眠（否则笔记本一睡训练就冻住，白等几小时）。
    实测本机"无人参与时系统睡眠超时"只有 5 分钟，必须自己按住。
    """
    try:
        import ctypes  # noqa: PLC0415

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 YOLO11（CPU）")
    ap.add_argument("--data", default=str(ROOT / "data.yaml"))
    ap.add_argument("--model", default=str(ROOT / "yolo11n.pt"),
                    help="预训练权重（迁移学习，强烈建议用；填 yolo11n.yaml 则是从零训练）")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640, help="想快点可先用 512")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="本机无 CUDA，保持 cpu")
    ap.add_argument("--workers", type=int, default=0, help="受限环境必须 0")
    ap.add_argument("--cache", action="store_true", help="把图片缓存进内存（RAM >= 8GB 空闲时用）")
    ap.add_argument("--patience", type=int, default=20, help="多少 epoch 没提升就早停")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="yolo11n_cpu")
    ap.add_argument("--resume", action="store_true", help="断点续训（接 last.pt）")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_file():
        print(f"[错误] 找不到 {data}，请先运行 prepare_dataset.py")
        return 2
    if not Path(args.model).exists() and not args.model.endswith(".yaml"):
        print(f"[错误] 找不到权重 {args.model}")
        return 2

    # 训练前先粗查标签是否齐全，避免白跑几小时
    try:
        import yaml as _yaml  # noqa: PLC0415

        import validate_labels as vl  # noqa: PLC0415
        cfg = _yaml.safe_load(data.read_text(encoding="utf-8")) or {}
        ds = Path(cfg.get("path", data.parent / "dataset"))
        if not ds.is_dir():
            ds = data.parent / "dataset"
        n_missing = 0
        for split in ("train", "val"):
            res = vl.check_split(ds / "images" / split, ds / "labels" / split, vl.CLASSES)
            if res["missing"]:
                n_missing += len(res["missing"])
                print(f"[警告] {split} 有 {len(res['missing'])} 张图还没有标签，"
                      "这些图会被当成背景图训练！")
        if n_missing:
            print("       建议先跑 validate_labels.py 确认标注是否漏了。")
            if input(f"仍有 {n_missing} 张缺标签，仍然继续训练？(y/N) ").strip().lower() != "y":
                return 1
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"[提示] 标签预检跳过：{exc}")

    from ultralytics import YOLO  # noqa: PLC0415

    if keep_awake():
        print("[提示] 已阻止系统睡眠，训练期间电脑不会睡（笔记本请插电源）")

    print("=" * 78)
    print(f"模型 {args.model} | 数据 {data} | epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch} device={args.device} workers={args.workers} cache={args.cache}")
    print("=" * 78)

    model = YOLO(args.model)
    t0 = time.time()
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        patience=args.patience,
        seed=args.seed,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        pretrained=args.model.endswith(".pt"),
        plots=True,
        val=True,
        verbose=True,
    )
    mins = (time.time() - t0) / 60

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print("=" * 78)
    print(f"训练结束，用时 {mins:.1f} 分钟（{mins / 60:.2f} 小时）")
    print(f"结果目录：{save_dir}")
    print(f"最佳权重：{best}   存在：{best.is_file()}")
    print("要提交的截图：results.png（loss/P/R/mAP 曲线）、confusion_matrix.png、"
          "val_batch*.jpg、以及上面的终端输出")

    if best.is_file():
        print("\n用最佳权重复核一次验证集指标：")
        YOLO(str(best)).val(data=str(data), imgsz=args.imgsz, device=args.device,
                            workers=args.workers, project=str(ROOT / "runs" / "val"),
                            name=f"{args.name}_val", exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
