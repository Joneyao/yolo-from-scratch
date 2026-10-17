"""第 5 篇：评估脚本，对比多个模型在同一份数据上的 mAP。

用法：
    python evaluate.py --data ../04_dataset_train/data.yaml
"""
import argparse

from ultralytics import YOLO

# 想对比的模型，改成你要比的
MODELS = ["yolov8n.pt", "yolov8s.pt"]


def main() -> None:
    parser = argparse.ArgumentParser(description="对比多个模型的 mAP")
    parser.add_argument("--data", default="data.yaml", help="数据集配置文件")
    args = parser.parse_args()

    for name in MODELS:
        model = YOLO(name)
        metrics = model.val(data=args.data)
        # map50 = mAP@0.5，map = mAP@0.5:0.95
        print(f"{name}: mAP@0.5={metrics.box.map50:.3f}  mAP@0.5:0.95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
