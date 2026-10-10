"""本地图片识别页面（Level 5）—— 上传图片 → 自己训练的 YOLO 模型推理 → 展示结果。

启动：
    streamlit run app.py
然后浏览器打开 http://localhost:8501

页面流程：
    选择/上传图片 → 后端加载 runs/exp1/weights/best.pt → 推理 → 并排显示原图与结果图，
    并列出每个目标的类别、置信度和位置。
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/exp1/weights/best.pt"
NAMES = {0: "obstacle", 1: "cola", 2: "football"}
COLORS = {0: (255, 170, 40), 1: (60, 60, 230), 2: (60, 220, 60)}   # BGR

st.set_page_config(page_title="机器人场景目标检测", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="正在加载模型……")
def load_model(path: str) -> YOLO:
    """模型只加载一次，之后复用（Streamlit 每次交互都会重跑脚本）。"""
    return YOLO(path)


def draw(img: np.ndarray, boxes) -> np.ndarray:
    out = img.copy()
    for box in boxes:
        cid = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        color = COLORS.get(cid, (255, 255, 255))
        text = f"{NAMES.get(cid, cid)} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, max(th, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


st.title("机器人场景目标检测")
st.caption("基于自训练 YOLOv8 模型，识别 obstacle（蓝色障碍箱）/ cola（可乐瓶）/ football（足球）")

with st.sidebar:
    st.header("设置")
    weights = st.text_input("模型权重", DEFAULT_WEIGHTS)
    conf = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.select_slider("推理尺寸", options=[320, 416, 512, 640, 768], value=640)
    st.divider()
    st.write("**使用步骤**")
    st.write("1. 上传一张本地图片")
    st.write("2. 等待模型推理")
    st.write("3. 查看检测结果与目标列表")

if not os.path.exists(weights):
    st.error(f"找不到权重文件：{weights}\n\n请先运行 `python train.py` 完成训练。")
    st.stop()

model = load_model(weights)

SAMPLE_DIR = "new_images"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

up = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png", "bmp", "webp"])

samples = []
if os.path.isdir(SAMPLE_DIR):
    samples = sorted(n for n in os.listdir(SAMPLE_DIR) if n.lower().endswith(EXTS))
sample = None
if up is None and samples:
    # 默认就选中第一张示例图，打开页面立刻能看到一次完整检测
    sample = st.selectbox("没有现成图片？也可以直接选一张示例图片试试",
                          ["（不选）"] + samples, index=1)

img = None
src_name = ""
if up is not None:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    src_name = up.name
elif sample and sample != "（不选）":
    img = cv2.imread(os.path.join(SAMPLE_DIR, sample))
    src_name = sample
else:
    st.info(f"请在上方上传图片。也可以把图片放进 `{SAMPLE_DIR}/` 目录，用 predict.py 批量检测。")
    st.stop()

if img is None:
    st.error("图片解码失败，请换一张试试。")
    st.stop()

st.caption(f"当前图片：{src_name}")

res = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
drawn = draw(img, res.boxes)

left, right = st.columns(2)
with left:
    st.subheader("原始图片")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
with right:
    st.subheader("检测结果")
    st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), use_container_width=True)

st.subheader(f"识别到的目标（共 {len(res.boxes)} 个）")
if len(res.boxes) == 0:
    st.warning("没有检测到目标。可以试着把左侧的置信度阈值调低一些。")
else:
    rows = []
    for box in res.boxes:
        cid = int(box.cls.item())
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        rows.append({
            "类别": NAMES.get(cid, cid),
            "置信度": round(float(box.conf.item()), 4),
            "左上角": f"({x1}, {y1})",
            "右下角": f"({x2}, {y2})",
            "框宽": x2 - x1,
            "框高": y2 - y1,
        })
    df = pd.DataFrame(rows).sort_values("置信度", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.groupby("类别")["置信度"].max())
