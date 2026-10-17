"""第 2 篇实操：打印 YOLOv8 检测头输出张量维度。

用法：
    pip install ultralytics
    python output_tensor.py
"""
import torch
from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")
    x = torch.zeros(1, 3, 640, 640)   # 造一张 640×640 的空图
    with torch.no_grad():
        out = model.model(x)          # 拿到检测头原始输出
    # out[0] 是展平后的预测：(1, 84, 8400)
    # 84 = 4 框坐标 + 80 类别；8400 = 80×80 + 40×40 + 20×20
    print("展平输出 shape:", tuple(out[0].shape))


if __name__ == "__main__":
    main()
