"""检查训练集和验证集的 YOLO 标签格式及完整性。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "dataset"
VALID_CLASSES = {0, 1, 2}


def main() -> None:
    errors: list[str] = []
    checked_boxes = 0
    for split in ("train", "val"):
        images = list((ROOT / "images" / split).glob("*"))
        for image in images:
            label = ROOT / "labels" / split / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"缺少标签：{label}")
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label}:{line_number} 应有 5 个字段")
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{label}:{line_number} 含非数字字段")
                    continue
                if class_id not in VALID_CLASSES:
                    errors.append(f"{label}:{line_number} 类别编号应为 0、1 或 2")
                if not all(0.0 <= value <= 1.0 for value in values):
                    errors.append(f"{label}:{line_number} 坐标必须在 0 到 1 之间")
                if values[2] <= 0 or values[3] <= 0:
                    errors.append(f"{label}:{line_number} 边界框宽高必须大于 0")
                checked_boxes += 1
    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"检查失败，共 {len(errors)} 个问题")
    print(f"检查通过，共检查 {checked_boxes} 个目标框。")


if __name__ == "__main__":
    main()

