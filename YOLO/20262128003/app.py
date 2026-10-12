"""本地图片目标检测页面（Level 5）。

启动方式（不能用 python app.py，必须用 streamlit 启动服务器）：

    streamlit run app.py

页面流程：上传图片 → 加载自己训练的 best.pt → 推理 → 显示原图、检测结果图、类别与置信度。
"""
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "yolo11n_3classes" / "weights" / "best.pt"


@st.cache_resource
def load_model(path: str) -> YOLO:
    """加载自己训练得到的权重。

    Streamlit 每次交互都会从头重跑整个脚本，不加缓存就会反复读取 5.2 MB 权重，
    页面会非常卡。@st.cache_resource 让模型只在首次运行时加载一次。
    """
    return YOLO(path)


def main() -> None:
    st.set_page_config(page_title="YOLO 机器人场景目标检测", layout="wide")
    st.title("基于 YOLO 的机器人场景目标检测")
    st.caption("检测类别：obstacle 障碍物 / cola 可乐罐 / football 足球")

    if not WEIGHTS.exists():
        st.error(f"未找到自己训练的模型权重：{WEIGHTS}")
        return

    model = load_model(str(WEIGHTS))
    # 把实际使用的权重路径显示出来，证明调用的是本项目自己训练的模型
    st.write(f"当前使用的模型（本项目自己训练得到）：`{WEIGHTS.relative_to(ROOT)}`")

    conf = st.slider("置信度阈值 conf", 0.05, 0.95, 0.25, 0.05)

    uploaded = st.file_uploader("选择一张本地图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("请上传一张图片开始检测。")
        return

    image = Image.open(uploaded).convert("RGB")

    with st.spinner("模型推理中……"):
        # 传 PIL 对象而不是 numpy 数组：ultralytics 对裸 numpy 数组按 BGR 解释，
        # 传 PIL 才会按 RGB 正确处理，不会出现颜色通道颠倒。
        result = model.predict(image, conf=conf, imgsz=640, device=0, verbose=False)[0]

    # result.plot() 返回 OpenCV 风格的 BGR 图像，转成 RGB 才能在页面上正确显示
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    col_left, col_right = st.columns(2)
    col_left.image(image, caption="原始图片")
    col_right.image(annotated, caption="检测结果")

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        st.warning("未检测到目标。可以试着调低左侧的置信度阈值再试。")
        return

    rows = []
    for (x1, y1, x2, y2), score, cls in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    ):
        rows.append({
            "类别": result.names[int(cls)],
            "置信度": round(float(score), 4),
            "左上角 x,y": f"{int(x1)}, {int(y1)}",
            "右下角 x,y": f"{int(x2)}, {int(y2)}",
        })

    st.subheader(f"共检测到 {len(rows)} 个目标")
    st.dataframe(rows)


if __name__ == "__main__":
    main()
