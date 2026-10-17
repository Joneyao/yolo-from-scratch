"""第 4 篇：训练脚本。

用法：
    python train.py --data data.yaml --model yolov8n.pt --epochs 100
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="训练一个 YOLOv8 检测模型")
    parser.add_argument("--data", default="data.yaml", help="数据集配置文件")
    parser.add_argument("--model", default="yolov8n.pt", help="初始权重")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小，显存不足就调小")
    parser.add_argument("--device", default="0", help="GPU 编号，CPU 写 cpu")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
