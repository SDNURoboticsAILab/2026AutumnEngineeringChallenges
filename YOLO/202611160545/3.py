import json, random, shutil
from pathlib import Path

SRC = r"D:\YOLO-Challenge\to_label"
JSON = r"D:\YOLO-Challenge\标注结果.json"
OUT = r"D:\YOLO-Challenge\dataset"
VAL_RATIO = 0.2

def main():
    data = json.loads(Path(JSON).read_text(encoding="utf-8"))
    imgs = data.get("images", {})
    names = sorted(imgs.keys())

    random.seed(0)
    shuffled = names[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    val_set = set(shuffled[:n_val])

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (Path(OUT) / sub).mkdir(parents=True, exist_ok=True)

    for name in names:
        src = Path(SRC) / name
        if not src.exists():
            continue
        split = "val" if name in val_set else "train"
        shutil.copy2(src, Path(OUT) / "images" / split / name)
        lines = []
        for o in imgs[name]:
            lines.append(f"{int(o['c'])} {o['x']:.6f} {o['y']:.6f} {o['w']:.6f} {o['h']:.6f}")
        (Path(OUT) / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

    yaml_text = (
        f"path: {Path(OUT).as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: obstacle\n"
        "  1: cola\n"
        "  2: football\n"
    )
    (Path(OUT).parent / "数据集配置.yaml").write_text(yaml_text, encoding="utf-8")
    print("done")

if __name__ == "__main__":
    main()
