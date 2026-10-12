from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    weights = root / "runs" / "yolo11n_3classes" / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"未找到训练权重：{weights}")
    model = YOLO(weights)
    model.predict(
        source=root / "dataset" / "images" / "test",
        conf=0.25,
        imgsz=640,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=root / "results",
        name="test_predictions",
    )


if __name__ == "__main__":
    main()

