from ultralytics import YOLO

if __name__ == '__main__':
    model=YOLO(r"yolov8n.pt")
    model.train(
        data=r"D:\大学四年\代码文件\deep learning\ultralytics-8.3.163\ultralytics\cfg\datasets\robot-plus.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        cache=False,
        workers=1,
        project="results",
    )
