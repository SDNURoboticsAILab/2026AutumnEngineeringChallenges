from ultralytics import YOLO

model = YOLO(r"runs\detect\runs\train\exp\weights\best.pt")
model.predict(
    source=r"predict_jpg/3",
    save=True,
    show=False,
    save_txt=True,
)