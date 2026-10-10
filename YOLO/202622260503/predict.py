from ultralytics import YOLO

model = YOLO(r"D:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\datasets\yolo_project\runs\detect\train\weights\best.pt")
model.predict(
    source=r"D:\deeplearning\make_dataset\rawimages",
    save=True,
    show=False,

)