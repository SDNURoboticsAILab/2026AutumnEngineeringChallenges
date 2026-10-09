from ultralytics import YOLO

model=YOLO(r"D:\大学四年\代码文件\deep learning\ultralytics-8.3.163\runs\detect\train2\weights\best.pt")
model.predict(
    source=r"D:\大学四年\代码文件\deep learning\make dataset\images",
save=True,
show=False,
    save_txt=True,
)