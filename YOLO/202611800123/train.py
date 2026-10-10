from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=416,
        batch=4,
        workers=0,
        device="cpu",
        project="runs/train",
        name="exp",
        patience=15,
        seed=42
    )