import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO 机器人场景目标检测", layout="wide")
st.title("🤖 基于 YOLO 的机器人场景目标检测")
st.write("上传一张图片，系统将自动检测图中的 `obstacle`、`cola`、`football`。")

# 1. 加载模型
@st.cache_resource
def load_model():
    # 把路径替换为你自己的 best.pt 路径
    return YOLO("runs/detect/my_robot_model/weights/best.pt")

model = load_model()

# 2. 图片上传
uploaded_file = st.file_uploader("请选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示原始图片
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图片")
        st.image(image, use_column_width=True)
        
    # 3. 进行推理
    with st.spinner("模型检测中..."):
        results = model(image)
        annotated_img = results[0].plot()  # 画好框的图
        
    # 4. 显示检测结果
    with col2:
        st.subheader("检测结果")
        st.image(annotated_img, channels="BGR", use_column_width=True)
        
    # 5. 显示识别出的目标类别和置信度
    st.subheader("📊 识别详情")
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.write("未检测到目标。")
    else:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            st.write(f"- **类别:** {cls_name}, **置信度:** {conf:.2f}")