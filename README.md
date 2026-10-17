# yolo-from-scratch

「从零开始学 YOLO」系列（公众号【轻松学AI】）的配套代码仓库。

公众号正文只贴核心片段，完整可运行代码都在这里，按篇组织：

- `01_predict/` — 第 1 篇：用预训练权重推理一张图
- `02_output_tensor/` — 第 2 篇：可视化模型输出张量
- `03_architecture/` — 第 3 篇：打印每层特征图 shape
- `04_dataset_train/` — 第 4 篇：数据集制作 + 训练全流程
- `05_evaluation/` — 第 5 篇：评估与调优
- `06_export_deploy/` — 第 6 篇：导出与部署
- `extra_01_predict_code/` — 番外 1：predict.py 推理链路精读
- `extra_02_pitfalls/` — 番外 2：踩坑手册
- `extra_03_selection/` — 番外 3：YOLOv8/v10 选型

## 协议说明

- 本仓库代码（脚本 / notebook）采用 **MIT** 协议。
- 依赖 [Ultralytics](https://github.com/ultralytics/ultralytics)（**AGPL-3.0**），作为 pip 依赖安装使用；本仓库**不复制、不修改** Ultralytics 源码。

## 快速开始

```bash
pip install ultralytics
python 01_predict/predict.py
```
