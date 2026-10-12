from pathlib import Path

from ultralytics import YOLO

# 用学号作为固定随机种子，保证训练结果可复现。
# 注意：numpy 要求种子落在 0 ~ 2**32-1 之间，直接用 20262128003 会抛出
# ValueError: Seed must be between 0 and 2**32 - 1，因此这里先取模。
SEED = 20262128003 % (2**32)


def main() -> None:
    root = Path(__file__).resolve().parent
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(root / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project=str(root / "runs"),
        name="yolo11n_3classes",
        patience=20,
        seed=SEED,
    )


if __name__ == "__main__":
    main()

