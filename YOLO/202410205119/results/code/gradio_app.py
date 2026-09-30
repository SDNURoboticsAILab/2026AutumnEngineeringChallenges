#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gradio_app.py —— Level 5：本地网页图片识别页面（Gradio）
================================================================================
位置: 你的学号/gradio_app.py

【实现的功能】
    打开网页 → 上传一张图片 → 后端用【自己训练的 best.pt】推理 →
    页面同时显示「原图」和「检测结果图」，并用表格列出识别到的类别与置信度。

【Windows 一键执行命令】（在本文件所在目录下执行）
    :: 1) 安装 Gradio（只需一次）
    pip install gradio

    :: 2) 启动网页
    python gradio_app.py

    :: 3) 浏览器打开终端提示的地址，默认 http://127.0.0.1:7860

    :: 其他常用参数
    python gradio_app.py --port 7861            # 换端口
    python gradio_app.py --host 0.0.0.0         # 允许同一局域网内手机访问
    python gradio_app.py --share                # 生成临时公网链接（Colab 常用）

【如果启动时报 PermissionError / WinError 5】
    某些受限环境禁止创建命名管道，而 Ultralytics 建标签缓存时会用到
    multiprocessing 的 ThreadPool。此时改用项目内的启动器（参数完全一致）：
        python run_gradio_shim.py

【注意】本页面加载的是【训练得到的权重】，不是官方预训练权重。
        若尚未训练，请先执行 python train.py。
"""

from __future__ import annotations

import argparse          # 命令行参数
import os                # 路径与环境变量
import sys               # 退出码
from pathlib import Path # 路径操作


# ---------------------------------------------------------------------------
# 0. 路径与常量
# ---------------------------------------------------------------------------
# 本文件已收纳到 results/code/ 下，因此项目根目录需要上溯两级：
# results/code/gradio_app.py -> results/code -> results -> 项目根（你的学号/）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 训练产物目录，best.pt 就在这里
RUNS_DETECT = PROJECT_ROOT / "runs" / "detect"

# 类别英文名 -> 中文名，用于页面显示
CLASS_ZH = {"obstacle": "障碍物", "cola": "可乐", "football": "足球"}

# 每类固定颜色（RGB），与 predict.py 保持一致，便于对照
PALETTE = {
    0: (255, 99, 71),     # obstacle 番茄红
    1: (30, 144, 255),    # cola     道奇蓝
    2: (255, 215, 0),     # football 金色
}

# 中文字体候选路径（找到第一个可用的就用）
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]


# ---------------------------------------------------------------------------
# 1. 权重定位
# ---------------------------------------------------------------------------
def find_best_weights(explicit: str = "") -> Path:
    """确定要加载的权重文件：显式传入优先，否则自动搜索最新的 best.pt。"""
    if explicit:
        w = Path(explicit)
        # 相对路径按项目根目录解析
        if not w.is_absolute():
            w = (PROJECT_ROOT / w).resolve()
        if not w.is_file():
            print(f"[X] 指定的权重不存在: {w}")
            sys.exit(1)
        return w

    cands = list(RUNS_DETECT.glob("*/weights/best.pt")) or \
        list(PROJECT_ROOT.glob("**/best.pt"))
    if not cands:
        print("[X] 没有找到任何 best.pt。")
        print("    请先训练模型: python train.py --epochs 60 --imgsz 320 --batch 2 --device cpu")
        sys.exit(1)
    # 多个候选取修改时间最新的
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


# ---------------------------------------------------------------------------
# 2. 画框（中文标签用 PIL 渲染，理由同 predict.py）
# ---------------------------------------------------------------------------
def load_zh_font(size: int = 18):
    """尝试加载一个中文字体；失败返回 (None, False)。"""
    try:
        from PIL import ImageFont
        for p in FONT_CANDIDATES:
            if os.path.isfile(p):
                try:
                    return ImageFont.truetype(p, size), True
                except Exception:
                    continue
    except Exception:
        pass
    return None, False


def draw_detections(pil_img, boxes, names):
    """在 PIL 图片上画检测框 + 类别名 + 置信度。

    参数:
        pil_img : PIL.Image（RGB）
        boxes   : list[(x1, y1, x2, y2, conf, cls_id)]
        names   : {cls_id: name}
    返回:
        (画好框的图片, 检测明细列表)
        明细列表形如 [{"类别": "可乐(cola)", "类别编号": 1, "置信度": 0.87,
                       "位置(x1,y1,x2,y2)": "150, 200, 292, 420"}]
        这个列表直接喂给 Gradio 的 Dataframe 组件显示。
    """
    from PIL import ImageDraw

    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font, zh_ok = load_zh_font(18)
    detections = []                       # 收集明细

    for (x1, y1, x2, y2, conf, cls_id) in boxes:
        cid = int(cls_id)
        color = PALETTE.get(cid, (0, 255, 0))

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 组织标签文字：有中文字体就"中文(英文) 置信度"，否则只有英文
        en = str(names.get(cid, f"class{cid}"))
        zh = CLASS_ZH.get(en, "")
        label = f"{zh}({en}) {conf:.2f}" if (zh_ok and zh) else f"{en} {conf:.2f}"

        # 画标签底色 + 文字
        if zh_ok:
            try:
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                tw, th = 9 * len(label), 18
            ty = y1 - th - 6                 # 默认画在框上方
            if ty < 0:                       # 贴顶边则画到框内
                ty = y1 + 3
            draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=color)
            draw.text((x1 + 4, ty + 2), label, fill=(0, 0, 0), font=font)
        else:
            # 无中文字体：用 OpenCV 画英文，保证页面不会因字体问题报错
            import cv2
            import numpy as np
            from PIL import Image

            arr = np.array(img)[:, :, ::-1].copy()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(y1 - th - 8, 0)
            cv2.rectangle(arr, (x1, ty), (x1 + tw + 8, ty + th + 8), color, -1)
            cv2.putText(arr, label, (x1 + 4, ty + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            img = Image.fromarray(arr[:, :, ::-1])

        # 汇总成一行表格数据
        detections.append({
            "类别": f"{zh}({en})" if zh else en,
            "类别编号": cid,
            "置信度": round(float(conf), 3),
            "位置(x1,y1,x2,y2)": f"{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}",
        })

    return img, detections


# ---------------------------------------------------------------------------
# 3. 构建 Gradio 界面
# ---------------------------------------------------------------------------
def build_app(model, default_conf: float, imgsz: int):
    """构建并返回 Gradio Blocks 应用。"""
    import gradio as gr                 # 延迟导入，缺依赖时报错更友好

    def detect(image, conf_thres):
        """核心回调：上传图片 -> 推理 -> 返回 (原图, 结果图, 明细表, 摘要)。"""
        # 用户还没传图时的兜底
        if image is None:
            return None, None, [], "请先上传一张图片。"

        import numpy as np
        from PIL import Image

        # Gradio 传进来的可能是 numpy 数组，统一转成 PIL.Image 便于处理
        pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) \
            else image.convert("RGB")

        # 调模型推理。conf 来自页面滑块，用户可以实时调松紧。
        res = model.predict(source=pil, conf=float(conf_thres), iou=0.45,
                            imgsz=imgsz, verbose=False)
        r = res[0]

        # 把张量转成 numpy 再打包给画图函数
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy()
        packed = [(float(a), float(b), float(c), float(d), float(e), float(f))
                  for (a, b, c, d), e, f in zip(xyxy, confs, clses)]

        drawn, dets = draw_detections(pil, packed, r.names)

        # 摘要文字：让用户一眼看到识别到了什么
        if dets:
            from collections import Counter
            cnt = Counter(d["类别"] for d in dets)
            summary = f"检测到 {len(dets)} 个目标：" + "，".join(
                f"{k} × {v}" for k, v in cnt.items())
        else:
            summary = (f"未检测到目标（当前置信度阈值 {conf_thres:.2f}）。"
                       f"可以把滑块往左调低一点再试。")

        return pil, drawn, dets, summary

    # ---- 页面布局 ----
    with gr.Blocks(title="YOLOv8 机器人场景目标检测演示") as demo:
        gr.Markdown(
            """
            # 🤖 YOLOv8 机器人场景目标检测演示
            上传一张图片，模型会自动框出 **obstacle（障碍物）**、**cola（可乐）**、
            **football（足球）** 三类目标，并给出置信度。

            > 权重来自本项目自制数据集训练的 `best.pt`（YOLOv8-n）。
            """
        )

        with gr.Row():
            # 左栏：输入
            with gr.Column():
                img_in = gr.Image(label="上传图片（原图）", type="pil", height=380)
                conf_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=default_conf, step=0.05,
                    label="置信度阈值 (conf)",
                    info="调低 = 框更多但可能误检；调高 = 只保留很确定的目标",
                )
            # 右栏：输出
            with gr.Column():
                img_out = gr.Image(label="检测结果", type="pil", height=380)

        # 明细表：类别 + 置信度（Level 5 明确要求展示）
        table = gr.Dataframe(
            headers=["类别", "类别编号", "置信度", "位置(x1,y1,x2,y2)"],
            label="识别结果明细", interactive=False, wrap=True,
        )
        summary = gr.Textbox(label="检测摘要", interactive=False)

        # 事件绑定：点按钮、上传图片、或拖动滑块都会触发检测
        btn = gr.Button("开始检测", variant="primary")
        btn.click(detect, inputs=[img_in, conf_slider],
                  outputs=[img_in, img_out, table, summary])
        img_in.upload(detect, inputs=[img_in, conf_slider],
                      outputs=[img_in, img_out, table, summary])
        conf_slider.release(detect, inputs=[img_in, conf_slider],
                            outputs=[img_in, img_out, table, summary])

        gr.Markdown("---\n权重文件：`runs/detect/train/weights/best.pt`")

    return demo


# ---------------------------------------------------------------------------
# 4. 入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="YOLOv8 本地网页检测演示（Gradio）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default="", help="best.pt 路径；留空自动搜索")
    p.add_argument("--conf", type=float, default=0.25, help="页面滑块初始置信度阈值")
    p.add_argument("--imgsz", type=int, default=320, help="推理输入边长")
    p.add_argument("--host", default="127.0.0.1",
                   help="监听地址；0.0.0.0 表示允许局域网访问")
    p.add_argument("--port", type=int, default=7860, help="监听端口")
    p.add_argument("--share", action="store_true",
                   help="生成 Gradio 公网临时链接（Colab 上有用）")
    return p.parse_args()


def main() -> None:
    """主流程：加载权重 -> 构建页面 -> 启动服务。"""
    args = parse_args()
    os.chdir(PROJECT_ROOT)             # 统一切到项目根目录

    # ---- 1) 加载模型 ----
    weights = find_best_weights(args.weights)
    print("=" * 70)
    print(f" 加载权重 : {weights}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
    except ImportError:
        print("[X] 未安装 ultralytics，请执行: pip install -r requirements.txt")
        sys.exit(1)
    print(f" 模型类别 : {model.names}")
    print("=" * 70)

    # ---- 2) 检查 gradio 是否安装 ----
    try:
        import gradio as gr             # noqa: F401
        print(f" Gradio 版本: {gr.__version__}")
    except ImportError:
        print("[X] 未安装 gradio，请执行: pip install gradio")
        sys.exit(1)

    # ---- 3) 构建并启动页面 ----
    demo = build_app(model, args.conf, args.imgsz)
    print(f"\n 启动网页服务: http://{args.host}:{args.port}")
    print(" 按 Ctrl+C 可停止服务\n")
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        # 允许页面访问 runs 目录（结果图保存在那里）
        allowed_paths=[str(RUNS_DETECT)],
    )


if __name__ == "__main__":
    main()
