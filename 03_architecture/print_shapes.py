"""第 3 篇实操：打印骨干网关键层特征图 shape。

用法：
    pip install ultralytics
    python print_shapes.py
"""
import torch
from ultralytics import YOLO


def main() -> None:
    m = YOLO("yolov8n.pt").model.eval()
    x = torch.zeros(1, 3, 640, 640)   # 造一张 640×640 的图
    stages = {4: "P3 小物体", 6: "P4 中物体", 8: "P5 大物体", 9: "SPPF"}
    with torch.no_grad():
        for i, layer in enumerate(m.model):
            x = layer(x)
            if i in stages:
                print(f"{stages[i]:8s} 层{i}: {tuple(x.shape)}")
            if i >= 9:
                break   # 只看骨干网，Neck 需要跳连，这里停住


if __name__ == "__main__":
    main()
