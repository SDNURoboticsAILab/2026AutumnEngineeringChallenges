from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\yolotask\datasets\runs\detect\train-3\weights\best.pt')
    model.predict(source=r'D:\yolotask\datasets\newimages', save=True)