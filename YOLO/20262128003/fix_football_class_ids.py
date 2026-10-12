"""备份并修正 football 图片标签中的错误类别编号。"""
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / "reports" / "label_backup" / "football_wrong_class_before_fix"


def main() -> None:
    corrected_files = 0
    corrected_boxes = 0
    for split in ("train", "val"):
        for label in sorted((ROOT / "dataset" / "labels" / split).glob("football_*.txt")):
            lines = label.read_text(encoding="utf-8").splitlines()
            output = []
            changed = False
            for line in lines:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] != "2":
                    fields[0] = "2"
                    corrected_boxes += 1
                    changed = True
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
