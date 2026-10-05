# -*- coding: utf-8 -*-
"""Level 5：本地图片识别页面（Streamlit）

流程：上传图片 → 加载自训 YOLO 模型 → 推理 → 展示原图/结果图/类别+置信度表

启动:
  conda activate yolo
  streamlit run yolo_project/app.py
  (或双击 启动页面.bat)

浏览器自动打开 http://localhost:8501
"""
import io
import os

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "yolo_project", "runs", "fixed_v4", "weights", "best.pt")

st.set_page_config(page_title="机器人场景目标检测", page_icon="🤖", layout="wide")
st.title("🤖 基于 YOLO 的机器人场景目标检测")
st.caption("obstacle（蓝色盒子） / cola（可乐瓶） / football（足球）三类别检测 —— 模型为本地自训练权重")


@st.cache_resource
def load_model():
    """模型只加载一次，页面刷新不重复加载（cache_resource 的作用）"""
    return YOLO(MODEL_PATH)


with st.sidebar:
    st.header("设置")
    conf = st.slider("置信度阈值 conf", 0.05, 0.9, 0.25, 0.05,
                     help="调高→误检少但可能漏检；调低→框多但杂")
    st.write(f"模型权重：`{os.path.basename(MODEL_PATH)}`")
    st.write(f"所在实验：`{os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))}`")

uploaded = st.file_uploader("选择一张本地图片上传", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    model = load_model()
    results = model.predict(img, conf=conf, verbose=False)
    r = results[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("① 上传的原始图片")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("② 检测结果图片")
        # ultralytics 8.4: r.plot() 返回 BGR numpy 数组，转 RGB 给 streamlit
        st.image(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("③ 识别出的目标")
    if len(r.boxes) == 0:
        st.warning("未检测到任何目标，可尝试调低左侧置信度阈值。")
    else:
        import pandas as pd
        rows = []
        for b in r.boxes:
            rows.append({
                "类别": model.names[int(b.cls)],
                "置信度": f"{float(b.conf):.1%}",
                "位置(cx,cy,w,h 归一化)": "(%.2f, %.2f, %.2f, %.2f)" % tuple(b.xywhn[0].tolist()),
            })
        df = pd.DataFrame(rows).sort_values("置信度", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.success(f"共检测到 {len(df)} 个目标：" +
                   "，".join(f"{c}×{n}" for c, n in df["类别"].value_counts().items()))
else:
    st.info("👆 请在左上角上传一张 jpg/png 图片开始检测。建议使用包含蓝色盒子/可乐瓶/足球场景的照片。")
