"""创建只包含尚未标注图片的 LabelImg 工作队列。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("class_name", choices=("obstacle", "cola", "football"))
    parser.add_argument("split", choices=("train", "val"))
    args = parser.parse_args()

    image_dir = ROOT / "dataset" / "images" / args.split
    label_dir = ROOT / "dataset" / "labels" / args.split
    queue_dir = ROOT / ".annotation_queue" / f"{args.class_name}_{args.split}"
    queue_dir.mkdir(parents=True, exist_ok=True)

    for old_file in queue_dir.iterdir():
        if old_file.is_file():
            old_file.unlink()

    remaining = []
    for image in sorted(image_dir.glob(f"{args.class_name}_*.*")):
        if not (label_dir / f"{image.stem}.txt").exists():
            target = queue_dir / image.name
            try:
                os.link(image, target)
            except OSError:
                target.write_bytes(image.read_bytes())
            remaining.append(image.name)

    print(f"{args.class_name} {args.split}: {len(remaining)} images remaining")


if __name__ == "__main__":
    main()
