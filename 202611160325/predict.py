from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的最优模型权重（直接读取当前目录下的 best.pt）
    model = YOLO('best.pt')
    # 对 dataset 文件夹里的图片进行预测
    model.predict(source='dataset', save=True)