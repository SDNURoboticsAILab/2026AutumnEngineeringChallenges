"""将原始 football 文件名标签映射回已划分数据集，并修正类别编号。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
YOLO_ROOT = ROOT.parent
DATASET = ROOT / "dataset"
SOURCE_IMAGES = YOLO_ROOT / "football"
SOURCE_LABELS = DATASET / "labels" / "train"
BACKUP = ROOT / "reports" / "label_backup" / "original_football_names"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def corrected_label(text: str) -> tuple[str, int]:
    corrected = 0
    output = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        # 本批标签来自 football 原始目录，人工画出的单目标框应属于 football（类别 2）。
        if fields[0] == "0":
            fields[0] = "2"
            corrected += 1
        output.append(" ".join(fields))
    return ("\n".join(output) + ("\n" if output else ""), corrected)


def main() -> None:
    image_map: dict[str, tuple[str, Path]] = {}
    for split in ("train", "val", "test"):
        for image in (DATASET / "images" / split).glob("football_*.*"):
            image_map[digest(image)] = (split, image)

    BACKUP.mkdir(parents=True, exist_ok=True)
    report = {
        "migrated": {"train": 0, "val": 0, "test": 0},
        "kept_existing": [],
        "corrected_class_0_to_2": 0,
        "unmatched": [],
    }
    for label in sorted(SOURCE_LABELS.glob("image_rgb_*.txt")):
        source_image = SOURCE_IMAGES / f"{label.stem}.jpg"
        if not source_image.exists():
            report["unmatched"].append(label.name)
            continue
        match = image_map.get(digest(source_image))
        if match is None:
            report["unmatched"].append(label.name)
            continue
        split, target_image = match
        target_label = DATASET / "labels" / split / f"{target_image.stem}.txt"
        text, corrected = corrected_label(label.read_text(encoding="utf-8"))
        report["corrected_class_0_to_2"] += corrected
        if target_label.exists():
            report["kept_existing"].append(target_label.relative_to(ROOT).as_posix())
        else:
            target_label.write_text(text, encoding="utf-8")
        report["migrated"][split] += 1
        shutil.move(str(label), BACKUP / label.name)

    classes_file = SOURCE_LABELS / "classes.txt"
    if classes_file.exists():
        shutil.move(str(classes_file), BACKUP / classes_file.name)
    report_path = ROOT / "reports" / "football_label_migration.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
