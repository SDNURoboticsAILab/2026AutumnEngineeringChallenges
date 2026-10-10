from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载模型（确保你下载了 yolov8n.pt 放在同级目录）
    model = YOLO("./yolov8n.pt")

    # 2. 开始训练
    model.train(
        data="data.yaml",       # 数据集配置文件
        epochs=10,              # 训练轮数（先设10跑通流程）
        imgsz=640,              
        batch=4,                
        device="cpu",           
        workers=0,              
        project="C:/Users/lenovo/Desktop/yolo-test/runs", 
        name="my_robot_model"   
    )

    print("训练完成！模型已保存在 runs/my_robot_model/weights/best.pt")