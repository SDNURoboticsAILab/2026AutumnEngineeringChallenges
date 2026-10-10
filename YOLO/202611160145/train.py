from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11n.pt")
    model.train(
        data=r"",
        epochs=1,
        imgsz=640,
        batch=30,
        cache=0,#相反cache="ram"为使用缓存，提前加载（要内存足够大）用于大像素图片最好
        workers=1,#同时的打包次数

    )
