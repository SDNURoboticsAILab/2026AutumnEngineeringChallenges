"""清理 dataset/labels 中按原始文件名保存的历史遗留标签。

背景：早期用 LabelImg 打开原始图片目录标注时，标签按原始文件名（image_rgb_*.txt）
保存在 dataset/labels/train 下。这些标签后来通过 sha256 比对迁移成了
obstacle_xxxx.txt / cola_xxxx.txt / football_xxxx.txt，但仍有少量文件残留在原处。

本脚本只有在同时满足下面三个条件时才把残留标签移入备份：
  1. 能在 obstacle / cola / football 原始目录中找到同名原图；
  2. 该原图能通过 sha256 唯一匹配到 dataset/images 中的某张图片；
  3. 目标标签文件已存在，且两者除类别编号外的边界框数值完全一致。

只要目标标签不存在就保留原文件并在报告中说明原因，绝不移动图片唯一的标注来源。
边界框与目标标签不一致时仍然移动，但会在报告中标记 geometry_matches_target=false，
方便人工复核：备份后原始内容仍可在 reports/label_backup/obsolete_original_names 找回。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
YOLO_ROOT = ROOT.parent
DATASET = ROOT / "dataset"
LABEL_DIRS = (DATASET / "labels" / "train", DATASET / "labels" / "val")
SOURCE_DIRS = ("obstacle", "cola", "football")
BACKUP = ROOT / "reports" / "label_backup" / "obsolete_original_names"
REPORT = ROOT / "reports" / "obsolete_label_cleanup.json"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def boxes(path: Path) -> list[tuple[float, ...]] | None:
    """读取标签文件，返回去掉类别编号后的边界框列表；格式异常时返回 None。"""
    values: list[tuple[float, ...]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            return None
        try:
            values.append(tuple(round(float(value), 6) for value in fields[1:]))
        except ValueError:
            return None
    return values


def build_image_index() -> dict[str, tuple[str, Path]]:
    index: dict[str, tuple[str, Path]] = {}
    for split in ("train", "val", "test"):
        for image in sorted((DATASET / "images" / split).glob("*.*")):
            index[digest(image)] = (split, image)
    return index


def find_source_image(stem: str) -> Path | None:
    for class_name in SOURCE_DIRS:
        candidate = YOLO_ROOT / class_name / f"{stem}.jpg"
        if candidate.exists():
            return candidate
    return None


def unique_destination(name: str) -> Path:
    target = BACKUP / name
    counter = 1
    while target.exists():
        target = BACKUP / f"{Path(name).stem}.{counter}{Path(name).suffix}"
        counter += 1
    return target


def main() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    image_index = build_image_index()

    moved: list[dict[str, str]] = []
    kept: list[dict[str, str]] = []
    classes_moved: list[str] = []

    for label_dir in LABEL_DIRS:
        if not label_dir.exists():
            continue
        for label in sorted(label_dir.glob("image_*.txt")):
            source_image = find_source_image(label.stem)
            if source_image is None:
                kept.append({"label": str(label.relative_to(ROOT)), "reason": "在原始目录中找不到同名原图"})
                continue
            match = image_index.get(digest(source_image))
            if match is None:
                kept.append({"label": str(label.relative_to(ROOT)), "reason": "原图未能匹配到 dataset/images 中的图片"})
                continue
            split, target_image = match
            target_label = DATASET / "labels" / split / f"{target_image.stem}.txt"
            if not target_label.exists():
                kept.append({
                    "label": str(label.relative_to(ROOT)),
                    "reason": f"目标标签 {target_label.relative_to(DATASET).as_posix()} 不存在，可能是唯一标注来源",
                })
                continue
            geometry_matches = boxes(label) == boxes(target_label)
            destination = unique_destination(label.name)
            shutil.move(str(label), destination)
            moved.append({
                "label": str(label.relative_to(ROOT)),
                "matched_image": f"{split}/{target_image.name}",
                "target_label": target_label.relative_to(DATASET).as_posix(),
                "geometry_matches_target": geometry_matches,
                "backup": str(destination.relative_to(ROOT)),
            })

        classes_file = label_dir / "classes.txt"
        if classes_file.exists():
            destination = unique_destination(f"classes_{label_dir.parent.parent.name}_{label_dir.name}.txt")
            shutil.move(str(classes_file), destination)
            classes_moved.append(str(destination.relative_to(ROOT)))

    report = {
        "moved_count": len(moved),
        "moved_geometry_mismatch_count": sum(1 for item in moved if not item["geometry_matches_target"]),
        "moved": moved,
        "kept_count": len(kept),
        "kept": kept,
        "classes_moved": classes_moved,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"移入备份：{len(moved)} 个残留标签")
    for item in moved[:5]:
        print(f"  {item['label']} -> 已有 {item['target_label']}")
    if len(moved) > 5:
        print(f"  ... 其余 {len(moved) - 5} 个见报告")
    mismatch = report["moved_geometry_mismatch_count"]
    if mismatch:
        print(f"注意：其中 {mismatch} 个与目标标签的边界框不完全一致，已按原样保留在备份目录，可人工复核。")
    print(f"保留待人工确认：{len(kept)} 个")
    for item in kept:
        print(f"  {item['label']}：{item['reason']}")
    if classes_moved:
        print(f"移动 labelImg 自动生成的 classes.txt：{len(classes_moved)} 个")
    print(f"报告：{REPORT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
