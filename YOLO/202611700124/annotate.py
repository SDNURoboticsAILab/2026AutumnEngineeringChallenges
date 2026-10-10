import cv2
import os
import numpy as np

# ================= 配置区 =================
# ⚠️ 请确保这些文件夹已经存在
IMAGE_DIR = "dataset/images/train"  # 待标注图片的存放路径
LABEL_DIR = "dataset/labels/train"  # 保存标签的路径

# 类别定义 (与你的 data.yaml 保持一致)
CLASSES = {0: 'obstacle', 1: 'cola', 2: 'football'}
COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)} # 红、绿、蓝
# =========================================

os.makedirs(LABEL_DIR, exist_ok=True)

# 全局变量
drawing = False
ix, iy = -1, -1
current_rect = None
img_original = None
img_display = None

def draw_rect(event, x, y, flags, param):
    """鼠标回调函数，用于画框"""
    global ix, iy, drawing, current_rect, img_display, img_original
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img_original.copy()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 255), 2)
            cv2.imshow("Annotator", img_display)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        current_rect = (x1, y1, x2 - x1, y2 - y1) # (x, y, w, h)
        img_display = img_original.copy()
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imshow("Annotator", img_display)

def main():
    global img_original, img_display, current_rect
    
    # 获取所有图片
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"❌ 在 {IMAGE_DIR} 中没有找到图片！")
        return

    print("="*40)
    print("🎯 简易 YOLO 标注工具")
    print("="*40)
    print("1. 鼠标左键拖拽绘制目标框")
    print("2. 按下数字键 0, 1, 2 分别对应 obstacle, cola, football")
    print("3. 按下 's' 保存当前图片的标签并进入下一张")
    print("4. 按下 'c' 清空当前图片的所有框")
    print("5. 按下 'n' 跳过当前图片")
    print("6. 按下 'q' 退出程序\n")

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotator", draw_rect)

    for img_name in img_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        # 使用 imdecode 解决中文路径读取报错问题
        img_original = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img_original is None:
            print(f"⚠️ 无法读取图片: {img_name}，跳过")
            continue
            
        img_h, img_w = img_original.shape[:2]
        boxes = [] # 存储当前图片的标注 (cls_id, x, y, w, h)
        current_rect = None

        while True:
            img_display = img_original.copy()
            
            # 绘制已经确认的框
            for box in boxes:
                cls_id, x, y, w, h = box
                color = COLORS.get(cls_id, (255, 255, 255))
                cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img_display, CLASSES.get(cls_id, str(cls_id)), (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制当前正在拖拽的框
            if current_rect is not None:
                x, y, w, h = current_rect
                cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 255), 2)

            cv2.imshow("Annotator", img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                print(f"跳过: {img_name}")
                break
            elif key == ord('c'):
                boxes = []
                current_rect = None
                print("🧹 已清空当前图片的框")
            elif key == ord('s'):
                # 保存 YOLO 格式标签
                label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
                with open(label_path, 'w', encoding='utf-8') as f:
                    for box in boxes:
                        cls_id, x, y, w, h = box
                        # YOLO 格式：类别 x中心 y中心 宽 高 (全部归一化到 0-1)
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                print(f"✅ 已保存: {label_path} (包含 {len(boxes)} 个目标)")
                break
            elif key in [ord('0'), ord('1'), ord('2')]:
                cls_id = int(chr(key))
                if current_rect is not None:
                    x, y, w, h = current_rect
                    if w > 0 and h > 0:
                        boxes.append((cls_id, x, y, w, h))
                        print(f" 添加框: {CLASSES[cls_id]}")
                    current_rect = None
                else:
                    print("⚠️ 请先用鼠标在图片上画框，然后再按数字键！")

    cv2.destroyAllWindows()
    print("\n🎉 标注完成！")

if __name__ == "__main__":
    main()