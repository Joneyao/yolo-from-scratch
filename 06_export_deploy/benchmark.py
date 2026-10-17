"""第 6 篇：推理速度基准（FPS）。

用法：
    python benchmark.py --model yolov8n.pt --n 100
"""
import argparse
import time

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="测量单张推理延迟与 FPS")
    parser.add_argument("--model", default="yolov8n.pt", help="权重")
    parser.add_argument("--n", type=int, default=100, help="推理次数（越多越准）")
    args = parser.parse_args()

    model = YOLO(args.model)
    x = torch.zeros(1, 3, 640, 640)  # 一张 640×640 的空图

    for _ in range(5):               # 预热，避免首次加载干扰
        model.predict(x, verbose=False)

    t0 = time.perf_counter()
    for _ in range(args.n):
        model.predict(x, verbose=False)
    ms = (time.perf_counter() - t0) / args.n * 1000
    print(f"单张推理约 {ms:.1f} ms，约 {1000 / ms:.0f} FPS")


if __name__ == "__main__":
    main()
