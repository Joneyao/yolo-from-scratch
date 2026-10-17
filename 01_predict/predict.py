"""第 1 篇实操脚本：用 yolov8n 预训练权重推理一张图。

用法：
    pip install ultralytics
    python predict.py
"""
from pathlib import Path
from urllib.request import urlretrieve

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main() -> None:
    bus = HERE / "bus.jpg"
    if not bus.exists():
        urlretrieve("https://ultralytics.com/images/bus.jpg", bus)

    model = YOLO("yolov8n.pt")  # 首次运行会自动下载权重
    results = model(str(bus))
    out = HERE / "bus_result.png"
    results[0].save(str(out))
    print(f"结果已保存：{out}")


if __name__ == "__main__":
    main()
