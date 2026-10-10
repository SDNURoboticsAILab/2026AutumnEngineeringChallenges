import os
import glob
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载你自己训练好的模型
    # ⚠️ 确保你把训练好的 best.pt 复制到了当前目录下
    model = YOLO("best.pt")
    
    # 2. 设置要检测的图片文件夹（推荐建一个 dataset/test 放新图片）
    source_dir = "dataset/images/val"  
    
    # 自动识别文件夹下的所有图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(source_dir, ext)))
        
    if not image_paths:
        print(f"⚠️ 在 {source_dir} 文件夹下没有找到任何图片！")
        exit()

    print(f"✅ 成功找到 {len(image_paths)} 张图片，开始自动检测...\n")
    
    os.makedirs("results", exist_ok=True)
    
    # 3. 逐张预测并手动绘制结果
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] 正在检测: {os.path.basename(img_path)}")
        
        # 进行推理
        results = model(img_path)
        img = cv2.imread(img_path) # 读取原始图片
        
        # 提取检测框信息
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            print("   -> 未检测到目标")
        
        # 遍历每一个检测到的目标，手动绘制
        for box in boxes:
            # 获取坐标 (左上角x, 左上角y, 右下角x, 右下角y)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # 获取类别ID和置信度
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            # 获取类别名称 (obstacle, cola, football)
            cls_name = model.names[cls_id]
            
            # 在终端打印详细信息 (方便截图)
            print(f"   -> 发现目标: {cls_name}, 置信度: {conf:.2f}")
            
            # === 开始绘图 (满足 Level 4 要求) ===
            # 1. 画目标框 (绿色框，线宽2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 2. 准备文字 (类别名称 + 置信度)
            label = f"{cls_name} {conf:.2f}"
            
            # 3. 计算文字背景框的大小
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # 4. 画文字背景框 (实心绿色，衬托白色文字)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            
            # 5. 写上类别名称和置信度 (白色文字)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        # 保存绘制好的图片
        img_name = os.path.basename(img_path)
        save_name = os.path.join("results", f"pred_{img_name}")
        cv2.imwrite(save_name, img)

    print(f"\n🎉 所有图片检测完成！请去 results 文件夹查看带有框和置信度的结果图。")