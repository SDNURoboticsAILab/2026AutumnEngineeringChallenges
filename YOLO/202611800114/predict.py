from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/runs/train/weights/best.pt")
    results = model.predict(
        source="test_images/",
        conf=0.25,
        save=True,
        project="results",
        name="predict",
        exist_ok=True,
    )
