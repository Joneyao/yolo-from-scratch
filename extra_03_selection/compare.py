"""番外 3：对比 YOLOv8 与 v10 的推理速度。

用法：
    python compare.py
"""
import time

import torch
from ultralytics import YOLO


def measure(model_name: str, n: int = 100) -> float:
    """返回单张推理耗时（毫秒）。"""
    model = YOLO(model_name)
    x = torch.zeros(1, 3, 640, 640)  # 一张 640×640 的空图
    for _ in range(5):               # 预热
        model.predict(x, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(x, verbose=False)
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    for name in ["yolov8n.pt", "yolov10n.pt"]:
        ms = measure(name)
        print(f"{name}: 单张约 {ms:.1f} ms，约 {1000 / ms:.0f} FPS")


if __name__ == "__main__":
    main()
