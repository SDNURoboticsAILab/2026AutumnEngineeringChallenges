#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py —— 加载 runs 里训练好的 best.pt，对新图片做目标检测并保存结果
================================================================================
位置: 你的学号/predict.py

【概念区分】
    训练(train) = 用标注好的图片调整模型权重，权重会改变；
    推理(infer) = 用训练好的权重做一次前向计算，权重不再改变。
    所以本脚本里不会出现 optimizer / loss.backward() 这类训练代码。

【Windows 一键执行命令】（在本文件所在目录下执行）
    :: 1) 检测验证集两张图（默认用法，权重自动从 runs 里找）
    python predict.py --source dataset/images/val

    :: 2) 检测单张图片
    python predict.py --source dataset/images/val/img05.jpg

    :: 3) 检测任意新图片目录（Level 4 要求用训练集之外的新图）
    python predict.py --source ..\\new_photos --conf 0.25 --print-dets

    :: 4) 显式指定权重 + 把结果同时复制到 results/ 便于提交
    python predict.py --weights runs/detect/train/weights/best.pt ^
                      --source dataset/images/val --copy-to-results

【输出约定（对齐仓库 --print-dets 输出格式）】
    终端会逐个目标打印：
        <图片名> | 类别=<类别名>(id=<编号>) 置信度=<0.000> 框=(x1,y1)-(x2,y2)
    结果图上同时呈现三要素（Level 4 验收要求）：
        ① 目标框   ② 类别名称   ③ 置信度
    文件输出：
        runs/detect/predict/<原文件名>.jpg    画好框的结果图
        results/predict/<原文件名>.jpg        加 --copy-to-results 时同步一份
        results/predict/detections.txt        加 --copy-to-results 时保存文本结果

【路径规范】只使用相对路径 + 基于 __file__ 的动态路径，无任何绝对路径。
"""

from __future__ import annotations

import argparse            # 命令行参数解析
import os                  # 工作目录切换
import shutil              # 复制结果文件到 results/
import sys                 # 退出码
from pathlib import Path   # 路径操作


# ---------------------------------------------------------------------------
# 0. 路径与常量
# ---------------------------------------------------------------------------
# 本文件所在目录就是项目根目录（你的学号/）
PROJECT_ROOT = Path(__file__).resolve().parent

# 训练产物所在目录。find_best_weights 会在 runs/detect/*/weights/ 下搜索
RUNS_DETECT = PROJECT_ROOT / "runs" / "detect"

# 推理结果要归档到的目录（提交用）
RESULTS_PREDICT = PROJECT_ROOT / "results" / "predict"

# 允许处理的图片后缀白名单
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 类别英文名 -> 中文显示名（画图时更直观，可用 --no-zh 关掉）
CLASS_ZH = {"obstacle": "障碍物", "cola": "可乐", "football": "足球"}

# 每个类别固定一种颜色，便于人眼区分（RGB 元组）
PALETTE = {
    0: (255, 99, 71),     # obstacle 番茄红
    1: (30, 144, 255),    # cola     道奇蓝
    2: (255, 215, 0),     # football 金色
}

# 中文字体候选：依次尝试 Windows / Linux / macOS 常见路径。
# 为什么需要：OpenCV 的 cv2.putText 只支持 ASCII，直接写中文会变成问号，
#             所以中文标签必须用 PIL 的 ImageFont 渲染。
# 一个都找不到时自动回退英文标签（功能不受影响）。
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",                                # Windows 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",                              # Windows 黑体
    r"C:\Windows\Fonts\simsun.ttc",                              # Windows 宋体
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",              # Linux 文泉驿
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",    # Linux Noto CJK
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",           # Linux 兜底
    "/System/Library/Fonts/PingFang.ttc",                        # macOS 苹方
]


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="加载训练好的 best.pt，对新图片做目标检测并保存画框结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --weights: 权重路径。留空则自动在 runs/detect/*/weights/ 下找 best.pt，
    #   命中多个时取【修改时间最新】的那个，省得记实验目录名。
    p.add_argument("--weights", default="",
                   help="模型权重路径；留空自动搜索 runs/detect/*/weights/best.pt")

    # --source: 待检测对象，可以是单张图片、图片目录，或图片 URL（必填）
    p.add_argument("--source", required=True,
                   help="待检测对象：单张图片 / 图片目录 / 图片URL")

    # --conf: 置信度阈值。低于该值的预测框直接丢弃。
    #   调低(0.10)：框更多但误检增加（Recall↑ Precision↓）
    #   调高(0.50)：只保留高把握的框（Precision↑ Recall↓）
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")

    # --iou: NMS（非极大值抑制）的 IoU 阈值，用于剔除同一目标上的重复框。
    #   两个框的重叠度超过该阈值时，只保留分数更高的那个。
    p.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")

    # --imgsz: 推理输入尺寸，建议与训练时保持一致（本项目训练用 320）
    p.add_argument("--imgsz", type=int, default=320, help="推理输入边长")

    # --device: 推理设备，"" = 自动；"cpu" 强制 CPU；"0" 用第一块 GPU
    p.add_argument("--device", default="", help="cpu / 0；留空为自动")

    # 结果输出位置
    p.add_argument("--project", default=str(RUNS_DETECT), help="结果输出根目录")
    p.add_argument("--name", default="predict", help="结果子目录名")
    p.add_argument("--exist-ok", action="store_true", help="允许覆盖同名输出目录")

    # --copy-to-results: 把结果图与文本结果额外复制到 results/predict/，便于提交
    p.add_argument("--copy-to-results", action="store_true",
                   help="把结果图与 detections.txt 复制到 results/predict/")

    # --save-txt: 额外保存 YOLO 格式的检测结果 txt（class cx cy w h）
    p.add_argument("--save-txt", action="store_true", help="同时保存 YOLO 格式 txt")
    p.add_argument("--save-conf", action="store_true", help="txt 中附带置信度")

    # 绘图相关
    p.add_argument("--line-width", type=int, default=2, help="检测框线宽")
    p.add_argument("--no-zh", action="store_true", help="标签只用英文类别名")

    # --print-dets: 逐个目标打印类别与置信度（对齐仓库输出格式）
    p.add_argument("--print-dets", action="store_true",
                   help="在终端打印每个检测框的类别与置信度")
    return p.parse_args()


def find_best_weights() -> Path:
    """自动搜索训练产出的 best.pt。

    搜索顺序：
      1) runs/detect/*/weights/best.pt（涵盖 train、train2… 所有实验）
      2) 兜底：项目内任意深度的 best.pt
    多个命中时按文件修改时间从新到旧排序，取最新的那个。
    找不到返回空 Path，由调用方给出中文提示。
    """
    candidates = list(RUNS_DETECT.glob("*/weights/best.pt"))
    if not candidates:
        candidates = list(PROJECT_ROOT.glob("**/best.pt"))
    if not candidates:
        return Path("")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_weights(arg_weights: str) -> Path:
    """确定最终使用的权重路径，缺失时给出明确的中文排错指引。"""
    if arg_weights:
        w = Path(arg_weights)
        # 相对路径按项目根目录解析，避免受当前工作目录影响
        if not w.is_absolute():
            w = (PROJECT_ROOT / w).resolve()
        if not w.is_file():
            print(f"[X] 找不到权重文件: {w}")
            print("    请先运行 python train.py 训练模型，或用 --weights 指定正确路径。")
            sys.exit(1)
        return w

    found = find_best_weights()
    if not found or not found.is_file():
        print("[X] 未在 runs/detect/*/weights/ 下找到 best.pt。")
        print("    本脚本要求使用【你自己训练得到的】权重，而不是官方预训练权重。")
        print("    请先执行: python train.py --epochs 60 --imgsz 320 --batch 2 --device cpu")
        sys.exit(1)
    print(f"[i] 自动找到权重: {found}")
    return found


def pick_font(size: int):
    """挑一个可用字体。

    返回 (font, ok)：
        ok=True  -> font 是 PIL FreeTypeFont，能渲染中文
        ok=False -> font 是 None，调用方应回退英文标签
    纯显示优化，失败不影响任何检测逻辑。
    """
    # 允许用环境变量强制关闭中文字体（排查字体相关异常时用）
    if os.environ.get("YOLO_NO_ZH_FONT") == "1":
        return None, False
    try:
        from PIL import ImageFont
        for path in FONT_CANDIDATES:
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size), True
                except Exception:
                    continue
    except Exception:
        pass
    return None, False


def draw_boxes_pil(image, boxes, names, line_width: int = 2, use_zh: bool = True):
    """用 PIL 在图片上绘制 检测框 + 类别名 + 置信度（支持中文）。

    参数:
        image : PIL.Image（RGB）
        boxes : list[(x1, y1, x2, y2, conf, cls_id)]
        names : {class_id: class_name}，直接取自模型，保证与训练时编号一致
    返回:
        画好框的新 PIL.Image
    """
    from PIL import ImageDraw

    img = image.copy()                       # 复制一份，不修改原图
    draw = ImageDraw.Draw(img)               # 创建画笔
    font, zh_ok = pick_font(16)              # 取中文字体（可能失败）
    use_pil_text = bool(zh_ok and use_zh)    # 决定用 PIL 还是 OpenCV 画文字

    for (x1, y1, x2, y2, conf, cls_id) in boxes:
        cid = int(cls_id)
        color = PALETTE.get(cid, (0, 255, 0))     # 该类别对应的颜色

        # ---- 1) 画矩形框：用多层偏移矩形模拟线宽，避免依赖 PIL 版本差异 ----
        for k in range(line_width):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)

        # ---- 2) 组织标签文字：类别名 + 置信度 ----
        en_name = str(names.get(cid, f"class{cid}"))
        if use_zh and en_name in CLASS_ZH:
            label = f"{CLASS_ZH[en_name]} {conf:.2f}"      # 例: 可乐 0.87
        else:
            label = f"{en_name} {conf:.2f}"                # 例: cola 0.87

        # ---- 3) 画标签底色块 + 文字 ----
        if use_pil_text:
            # 有中文字体：用 PIL 写，支持中文
            try:
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]      # 文字宽高
            except Exception:
                tw, th = 8 * len(label), 16                # 兜底估算
            ty = y1 - th - 4                               # 默认画在框上方
            if ty < 0:                                     # 框贴顶边时改画到框内
                ty = y1 + 2
            draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
            draw.text((x1 + 3, ty + 2), label, fill=(0, 0, 0), font=font)
        else:
            # 没有中文字体：回退到 OpenCV 画英文（不需要字体文件，必然可用）
            import cv2
            import numpy as np
            from PIL import Image

            arr = np.array(img)[:, :, ::-1].copy()         # PIL RGB -> OpenCV BGR
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y1 - th - 6
            if ty < 0:
                ty = y1 + 2
            cv2.rectangle(arr, (x1, ty), (x1 + tw + 4, ty + th + 6), color, -1)
            cv2.putText(arr, label, (x1 + 2, ty + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            img = Image.fromarray(arr[:, :, ::-1])         # BGR -> PIL RGB

    return img


def collect_sources(source: str) -> list[Path]:
    """把 --source 归一化成图片文件列表（支持单文件 / 目录 / URL）。"""
    s = str(source)
    # URL 交给 Ultralytics 自己下载，这里不做本地展开
    if s.startswith(("http://", "https://")):
        return []
    p = Path(s)
    if p.is_dir():
        # 目录则递归查找所有图片后缀的文件
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS)
    return [p]


def main() -> None:
    """主流程：确定权重 → 检查输入 → 推理 → 绘制并保存结果。"""
    args = parse_args()

    # 统一切到项目根目录，保证相对路径含义稳定
    os.chdir(PROJECT_ROOT)

    # =======================================================================
    # 1. 确定并加载权重
    # =======================================================================
    weights = resolve_weights(args.weights)
    print(f"[1/4] 加载模型权重: {weights}")

    # 延迟导入：缺依赖时给出中文提示而不是一长串报错栈
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[X] 未安装 ultralytics，请先执行: pip install -r requirements.txt")
        sys.exit(1)

    model = YOLO(str(weights))
    # model.names 是从权重文件里读出来的类别字典，形如
    # {0: 'obstacle', 1: 'cola', 2: 'football'}。
    # 直接用它而不是自己写死，可以保证画出来的名字和训练时的编号绝对一致。
    print(f"[i] 模型类别表: {model.names}")

    # =======================================================================
    # 2. 检查输入
    # =======================================================================
    src_path = Path(str(args.source))
    sources = collect_sources(args.source)
    if not str(args.source).startswith(("http://", "https://")):
        if src_path.is_dir():
            if not sources:
                print(f"[X] 目录里没有找到任何图片: {src_path}")
                sys.exit(1)
            print(f"[2/4] 输入是目录，共找到 {len(sources)} 张图片")
        elif not src_path.is_file():
            print(f"[X] 找不到输入图片: {src_path}")
            sys.exit(1)
        else:
            print(f"[2/4] 输入是单张图片: {src_path}")
    else:
        print(f"[2/4] 输入是网络图片: {args.source}")

    # =======================================================================
    # 3. 推理
    # =======================================================================
    print(f"[3/4] 开始推理（conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}）...")
    results = model.predict(
        source=args.source,           # 输入
        conf=args.conf,               # 置信度阈值
        iou=args.iou,                 # NMS 的 IoU 阈值
        imgsz=args.imgsz,             # 推理输入尺寸
        device=args.device or None,   # 推理设备
        save=True,                    # 保存画好框的结果图
        save_txt=args.save_txt,       # 可选：保存 YOLO 格式 txt
        save_conf=args.save_conf,     # 可选：txt 里附置信度
        project=args.project,         # 输出根目录
        name=args.name,               # 输出子目录名
        exist_ok=args.exist_ok,       # 允许覆盖
        line_width=args.line_width,   # 框线宽
        show_labels=True,             # 显示类别名
        show_conf=True,               # 显示置信度
        verbose=False,                # 关掉框架自带的逐张打印，我们自己打印
    )

    # =======================================================================
    # 4. 用 PIL 重绘（支持中文标签）+ 打印检测结果
    # =======================================================================
    print("[4/4] 绘制检测结果 ...")
    save_dir = Path(results[0].save_dir) if results else (RUNS_DETECT / args.name)
    total_boxes = 0                        # 统计总检测框数
    text_lines = []                        # 收集文本结果，用于写入 detections.txt

    for r in results:
        src_file = Path(r.path)                      # 本次推理的原图路径
        xyxy = r.boxes.xyxy.cpu().numpy()            # 所有框坐标 (x1,y1,x2,y2)
        confs = r.boxes.conf.cpu().numpy()           # 每个框的置信度
        clses = r.boxes.cls.cpu().numpy()            # 每个框的类别编号
        total_boxes += len(xyxy)

        # ---- 4a. 中文重绘：覆盖 Ultralytics 保存的英文结果图 ----
        try:
            from PIL import Image

            with Image.open(src_file) as im:
                pil_img = im.convert("RGB")
                # 把 numpy 数组整理成 draw_boxes_pil 需要的元组列表
                packed = [
                    (float(a), float(b), float(c), float(d), float(e), float(f))
                    for (a, b, c, d), e, f in zip(xyxy, confs, clses)
                ]
                drawn = draw_boxes_pil(pil_img, packed, r.names,
                                       line_width=args.line_width,
                                       use_zh=not args.no_zh)
                out_file = save_dir / src_file.name
                drawn.save(out_file)
                print(f"      已保存: {out_file}")
        except Exception as e:
            # 中文重绘只是显示优化，失败就保留英文版本，不影响检测结果本身
            print(f"      [!] 中文标签绘制失败（{e!r}），保留原始结果图。")

        # ---- 4b. 逐目标打印：对齐仓库要求的 --print-dets 输出格式 ----
        if len(xyxy) == 0:
            line = f"{src_file.name} | 未检测到目标（可尝试调低 --conf）"
            text_lines.append(line)
            if args.print_dets:
                print(f"      {line}")
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clses):
            name = r.names.get(int(k), f"class{int(k)}")
            line = (f"{src_file.name} | 类别={name}(id={int(k)}) "
                    f"置信度={float(c):.3f} 框=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
            text_lines.append(line)
            if args.print_dets:
                print(f"      {line}")

    print(f"\n[done] 共 {len(results)} 张图片、{total_boxes} 个检测框")
    print(f"       结果目录: {save_dir}")

    # =======================================================================
    # 5. 可选：把结果复制到 results/ 目录，便于整理提交材料
    # =======================================================================
    if args.copy_to_results:
        RESULTS_PREDICT.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src_file in sorted(save_dir.glob("*.jpg")):
            shutil.copy2(src_file, RESULTS_PREDICT / src_file.name)
            copied += 1
        # 文本结果也存一份
        txt_path = RESULTS_PREDICT / "detections.txt"
        txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        print(f"\n[结果归档] 已复制 {copied} 张结果图到 {RESULTS_PREDICT}")
        print(f"[结果归档] 文本结果已写入 {txt_path}")


# 只有直接运行本文件时才执行 main()
if __name__ == "__main__":
    main()
