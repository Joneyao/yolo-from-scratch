"""第 6 篇：导出脚本。

用法：
    python export.py --model yolov8n.pt --format onnx
    python export.py --model yolov8n.pt --format engine   # TensorRT，需 NVIDIA GPU
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 YOLO 模型")
    parser.add_argument("--model", default="yolov8n.pt", help="要导出的权重")
    parser.add_argument(
        "--format", default="onnx",
        help="onnx / engine / openvino / tflite / ncnn / torchscript 等",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    model.export(format=args.format)


if __name__ == "__main__":
    main()
