"""阶段 4：用自己训练的模型对新图片做目标检测（Level 4 —— 及格线）

关键点：
  * 权重必须是**自己训练**出来的 runs/train/<name>/weights/best.pt，
    绝不能用官方 yolo11n.pt 冒充（README 明确要求）。
  * 图片必须是**训练集之外**的：prepare_dataset.py 已经把每类预留的测试图放在
    D:\\YOLO\\test_images\\，直接用它即可。

用法：
    # 对预留的测试图整体推理
    D:\\YOLO\\.venv\\Scripts\\python.exe predict.py

    # 指定权重 / 指定图片（单张、目录、通配符都行）
    D:\\YOLO\\.venv\\Scripts\\python.exe predict.py --weights runs/train/yolo11n_cpu/weights/best.pt --source D:\\我的新照片

    # 想多留些低置信度目标看效果
    D:\\YOLO\\.venv\\Scripts\\python.exe predict.py --conf 0.15

结果：runs/predict/<name>/ 下有画好「目标框 + 类别名 + 置信度」的结果图，
      以及 detections.csv（每行：图片, 类别, 置信度, x1,y1,x2,y2）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gather(source: str) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
    return sorted(Path().glob(source))


def main() -> int:
    ap = argparse.ArgumentParser(description="新图片目标检测（Level 4）")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "train" / "yolo11n_cpu" / "weights" / "best.pt"))
    ap.add_argument("--source", default=str(ROOT / "test_images"), help="图片/目录/通配符")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="new_images")
    ap.add_argument("--max-images", type=int, default=0, help="只推理前 N 张（0=全部）")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"[错误] 找不到权重：{weights}")
        print("       请先训练：python train.py --epochs 60")
        return 2

    if "yolo11n.pt" in weights.name and "runs" not in str(weights):
        print("[警告] 你正在使用官方预训练权重，不是自己训练的模型！")
        print("       Level 4 要求必须用自己训练的 best.pt。")

    files = gather(args.source)
    if not files:
        print(f"[错误] 在 {args.source} 下没找到图片。")
        print("       训练集外的测试图应由 prepare_dataset.py 预留到 D:\\YOLO\\test_images\\")
        return 2
    if args.max_images:
        files = files[: args.max_images]
    print(f"待检测图片：{len(files)} 张（来自 {args.source}）")

    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights))
    names = model.names

    # 一次性批量推理：比逐张调用更快，也不会每张都刷一遍日志
    results = model.predict(
        source=[str(f) for f in files], conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device, save=True, project=str(ROOT / "runs" / "predict"),
        name=args.name, exist_ok=True, verbose=False,
    )

    rows: list[dict] = []
    n_hit = n_box = 0
    out_dir = ROOT / "runs" / "predict" / args.name
    for i, res in enumerate(results, 1):
        src = Path(res.path)
        out_dir = Path(res.save_dir)
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  [{i}/{len(files)}] {src.name}: 未检出目标")
            rows.append({"image": src.name, "class": "", "conf": "", "x1": "", "y1": "", "x2": "", "y2": ""})
            continue
        n_hit += 1
        n_box += len(boxes)
        print(f"  [{i}/{len(files)}] {src.name}: {len(boxes)} 个目标")
        for b in boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = (round(float(v), 1) for v in b.xyxy[0].tolist())
            cname = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
            print(f"        {cname:<10} conf={conf:.3f}  box=({x1},{y1})-({x2},{y2})")
            rows.append({"image": src.name, "class": cname, "conf": f"{conf:.3f}",
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    csv_path = out_dir / "detections.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["image", "class", "conf", "x1", "y1", "x2", "y2"])
            w.writeheader()
            w.writerows(rows)

    print("=" * 78)
    print(f"共 {len(files)} 张：{n_hit} 张有检出，合计 {n_box} 个目标")
    print(f"结果图目录：{out_dir}")
    print(f"检测明细表：{csv_path}")
    if n_box == 0:
        print("提示：一张都没检出。若模型刚训练完精度低，不代表代码有问题；"
              "也可能是置信度阈值偏高（可试 --conf 0.1）。")
    print("Level 4 提交要求：至少展示 ① 单个目标 ② 多个目标 ③ 不同场景/角度 三种情况，"
          "且图上能看到目标框、类别名、置信度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
