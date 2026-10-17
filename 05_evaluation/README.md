# 05_evaluation — 评估与调优

对应第 5 篇：对比不同模型 / 配置的 mAP。

```bash
pip install ultralytics
python evaluate.py --data ../04_dataset_train/data.yaml
```

重点看两行：`mAP@0.5` 和 `mAP@0.5:0.95`。
