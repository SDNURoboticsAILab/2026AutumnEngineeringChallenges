from ultralytics import YOLO

def main():
    model = YOLO(r"D:\YOLO-Challenge\runs\train\weights\best.pt")
    model.predict(
        source=r"D:\YOLO-Challenge\test_images",
        conf=0.25,
        save=True,
        save_txt=True,
        project=r"D:\YOLO-Challenge\runs",
        name="predict",
        exist_ok=True,
    )
    print("检测完成")

if __name__ == "__main__":
    main()
