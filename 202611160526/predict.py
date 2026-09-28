from ultralytics import YOLO

model = YOLO("runs/exp/weights/best.pt")
results = model.predict(
    source="dataset/images/val",
    save=True
)