"""备份并修正 cola 图片标签中误选的 obstacle 类别。"""
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / "reports" / "label_backup" / "cola_wrong_class_before_fix"


def main() -> None:
    corrected_files = 0
    corrected_boxes = 0
    for split in ("train", "val"):
        label_dir = ROOT / "dataset" / "labels" / split
        for label in sorted(label_dir.glob("cola_*.txt")):
            lines = label.read_text(encoding="utf-8").splitlines()
            output = []
            changed = False
            for line in lines:
                fields = line.split()
                if fields and fields[0] == "0":
                    fields[0] = "1"
                    changed = True
                    corrected_boxes += 1
                if fields:
                    output.append(" ".join(fields))
            if changed:
                destination = BACKUP / split / label.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label, destination)
                label.write_text("\n".join(output) + "\n", encoding="utf-8")
                corrected_files += 1
    print(f"Corrected {corrected_boxes} boxes in {corrected_files} files.")


if __name__ == "__main__":
    main()
