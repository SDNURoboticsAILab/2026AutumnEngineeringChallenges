import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. 设置网页标题和图标
st.set_page_config(page_title="YOLO 机器人检测系统", page_icon="🤖")
st.title("🤖 机器人实验室目标检测系统")
st.write("上传一张图片，系统将自动检测图中的可乐、足球和障碍物。")


# 2. 加载模型 (使用缓存，避免每次上传图片都重新加载模型)
@st.cache_resource
def load_model():
    # 加载你训练了100轮的模型
    return YOLO("runs/detect/train_v2/weights/best.pt")


model = load_model()

# 3. 创建文件上传组件
uploaded_file = st.file_uploader("请选择一张图片上传...", type=["jpg", "jpeg", "png"])

# 4. 如果用户上传了图片，就开始推理
if uploaded_file is not None:
    # 打开图片并显示
    image = Image.open(uploaded_file)
    st.image(image, caption="原始图片", width='stretch')  # 修复了旧参数

    with st.spinner("模型正在检测中，请稍候..."):
        # 调用 YOLO 模型进行预测
        results = model.predict(source=image, conf=0.25)

        # 获取画好框的图片
        annotated_image = results[0].plot()

        # 在网页上显示结果
        st.success("检测完成！")
        st.image(annotated_image, caption="检测结果", width='stretch')  # 修复了旧参数