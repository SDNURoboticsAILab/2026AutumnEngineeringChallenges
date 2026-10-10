"""把实验室提供的原始图片整理成 YOLO 数据集结构。

原始数据（实验室提供，位于仓库 YOLO/ 目录下）：
    obstacle/  316 张
    cola/      293 张
    football/  340 张

整理后：
    dataset/images/{train,val}/<类别>_<原名>.jpg
    dataset/labels/{train,val}/<类别>_<原名>.txt
    split_manifest.csv          # 划分清单，记录每张图去了 train 还是 val

划分方式：按类别内部独立随机划分，train:val = 8:2，随机种子固定为 42，
保证可复现，且三个类别在训练集/验证集中的比例一致。
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys

CLASSES = ["obstacle", "cola", "football"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42
VAL_RATIO = 0.2


def find_source(root: str, cls: str, explicit: str | None) -> str:
    """定位某一类别的原始图片目录。"""
    if explicit:
        return explicit
    for cand in (
        os.path.join(root, cls),                       # <repo>/YOLO/<cls>
        os.path.join(root, "YOLO", cls),
        os.path.join(root, "raw_data", cls),
    ):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"找不到类别 {cls} 的原始图片目录（尝试过 {root}/{cls} 等路径）")


def collect_images(src: str) -> list[str]:
    names = [
        n for n in sorted(os.listdir(src))
        if os.path.splitext(n)[1].lower() in EXTS and os.path.isfile(os.path.join(src, n))
    ]
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description="整理 obstacle/cola/football 为 YOLO 数据集")
    ap.add_argument("--src-root", default="..", help="原始数据根目录，默认上级目录（即仓库 YOLO/）")
    ap.add_argument("--out", default="dataset", help="数据集输出目录")
    ap.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--copy-images", action="store_true", default=True,
                    help="是否把图片复制进 dataset/images（训练需要）")
    ap.add_argument("--dry-run", action="store_true", help="只打印划分结果，不落盘")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    stats: dict[str, dict[str, int]] = {}

    for cls in CLASSES:
        src = find_source(args.src_root, cls, None)
        names = collect_images(src)
        if not names:
            print(f"[WARN] {cls}: 目录 {src} 下没有图片", file=sys.stderr)
            continue

        shuffled = names[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * args.val_ratio))
        val_names = set(shuffled[:n_val])

        counts = {"train": 0, "val": 0}
        for name in shuffled:
            split = "val" if name in val_names else "train"
            stem, ext = os.path.splitext(name)
            new_name = f"{cls}_{stem}{ext}"
            rows.append({
                "split": split, "class": cls,
                "source": os.path.join(src, name), "image": new_name,
            })
            counts[split] += 1

            if args.dry_run:
                continue

            img_dst = os.path.join(args.out, "images", split, new_name)
            if args.copy_images and not os.path.exists(img_dst):
                os.makedirs(os.path.dirname(img_dst), exist_ok=True)
                shutil.copy2(os.path.join(src, name), img_dst)
            # 标签文件先建同名空文件，下一步由预标注脚本写入内容
            lbl_dst = os.path.join(args.out, "labels", split, os.path.splitext(new_name)[0] + ".txt")
            os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
            if not os.path.exists(lbl_dst):
                open(lbl_dst, "w").close()

        stats[cls] = counts
        print(f"{cls:9s} 源={len(names):4d}  train={counts['train']:4d}  val={counts['val']:3d}   ({src})")

    total = {"train": sum(s["train"] for s in stats.values()),
             "val": sum(s["val"] for s in stats.values())}
    print(f"{'合计':9s} 源={total['train']+total['val']:4d}  train={total['train']:4d}  val={total['val']:3d}")

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        with open("split_manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split", "class", "image", "source"])
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["split"], r["class"], r["image"])))
        print("已写出 split_manifest.csv")
        for split in ("train", "val"):
            print(f"  dataset/images/{split}: {len(os.listdir(os.path.join(args.out,'images',split)))} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
