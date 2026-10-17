"""番外 1：逐段看懂推理链路。

把 model("bus.jpg") 拆成 preprocess / inference / postprocess 三段，
每段加注释，对照正文理解。
"""
import torch
from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")

    # ---------- 第 1 段：preprocess（预处理）----------
    # 真实实现会读图、resize 到 640、归一化到 [0,1]、拼成 batch。
    # 这里直接造一张归一化的 640×640 空图当输入。
    x = torch.zeros(1, 3, 640, 640)          # (batch, 通道, 高, 宽)
    print("preprocess 输出:", tuple(x.shape))

    # ---------- 第 2 段：inference（网络前向）----------
    with torch.no_grad():
        out = model.model(x)                  # 检测头原始输出
    print("inference 输出:", tuple(out[0].shape))   # (1, 84, 8400)

    # ---------- 第 3 段：postprocess（后处理）----------
    # 解码坐标、算置信度（物体置信度 × 类别得分）、跑 NMS。
    # 这里借用官方 predictor 完成后处理，再取结果里的框：
    result = model.predict("https://ultralytics.com/images/bus.jpg")[0]
    n = result.boxes.shape[0]
    print("postprocess 得到", n, "个框")
    # 每个框是 [x1, y1, x2, y2, 置信度, 类别id] 6 个数
    if n:
        print("第一个框:", result.boxes.data[0].tolist())


if __name__ == "__main__":
    main()
