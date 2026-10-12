"""检查原始图片，并按类别分层划分 train/val/test。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

CLASS_NAMES = ("obstacle", "cola", "football")
SEED = 20262128003
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_info(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or path.suffix.removeprefix(".").upper()
    return width, height, image_format


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_items(items: list[Path]) -> dict[str, list[Path]]:
    shuffled = items.copy()
    random.Random(SEED).shuffle(shuffled)
    train_end = round(len(shuffled) * 0.8)
    val_end = train_end + round(len(shuffled) * 0.1)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dataset")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    valid: dict[str, list[Path]] = {}
    hashes: dict[str, list[str]] = {}
    errors: list[str] = []

    for class_name in CLASS_NAMES:
        class_dir = args.source / class_name
        valid[class_name] = []
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                width, height, image_format = image_info(path)
                digest = sha256(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            valid[class_name].append(path)
            hashes.setdefault(digest, []).append(str(path))
            rows.append({
                "class": class_name,
                "filename": path.name,
                "width": width,
                "height": height,
                "format": image_format,
                "sha256": digest,
            })

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    if errors:
        raise RuntimeError("发现损坏图片：\n" + "\n".join(errors))

    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts: Counter[tuple[str, str]] = Counter()
    for class_name, paths in valid.items():
        for split, split_paths in split_items(paths).items():
            for index, source in enumerate(split_paths, start=1):
                target_name = f"{class_name}_{index:04d}{source.suffix.lower()}"
                shutil.copy2(source, args.output / "images" / split / target_name)
                split_counts[(class_name, split)] += 1

    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(exist_ok=True)
    with (report_dir / "image_inventory.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 数据检查与划分结果",
        "",
        f"- 随机种子：`{SEED}`",
        "- 划分比例：约 80% 训练、10% 验证、10% 最终测试",
        f"- 可读取图片：{len(rows)}",
        f"- 损坏图片：{len(errors)}",
        f"- 内容完全相同的重复图片组：{len(duplicate_groups)}",
        "",
        "| 类别 | train | val | test | 总数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in CLASS_NAMES:
        counts = [split_counts[(class_name, split)] for split in ("train", "val", "test")]
        lines.append(f"| {class_name} | {counts[0]} | {counts[1]} | {counts[2]} | {sum(counts)} |")
    lines.extend(["", "## 重复图片", ""])
    if duplicate_groups:
        for group in duplicate_groups:
            lines.append("- " + "；".join(group))
    else:
        lines.append("未发现内容完全相同的重复图片。")
    (report_dir / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()

