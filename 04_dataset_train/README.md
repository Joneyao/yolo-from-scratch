# 04_dataset_train — 数据集制作 + 训练

对应第 4 篇：自建小数据集训练一个 YOLO。

## 快速开始

1. 用 LabelStudio 标注图片，导出 YOLO 格式
2. 整理成 `images/` + `labels/` 的目录结构
3. 复制 `data.yaml.example` 为 `data.yaml`，改成你的路径和类别
4. 训练：

```bash
pip install ultralytics
python train.py --data data.yaml --model yolov8n.pt --epochs 100
```

## 标签格式

每张图对应一个同名 `.txt`，每行一个框：

```text
class_id  x_center  y_center  width  height
```

坐标归一化到 0~1，类别序号从 0 开始。
