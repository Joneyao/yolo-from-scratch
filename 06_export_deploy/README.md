# 06_export_deploy — 导出与部署

对应第 6 篇：导出 ONNX / TensorRT + 测 FPS。

```bash
pip install ultralytics
python export.py --model yolov8n.pt --format onnx      # 导出 ONNX
python export.py --model yolov8n.pt --format engine    # 导出 TensorRT（需 GPU）
python benchmark.py --model yolov8n.pt --n 100         # 测 FPS
```
