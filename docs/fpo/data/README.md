# Reported result snapshot

[reported-results.json](reported-results.json) contains the numerical results associated with the current manuscript (updated October 11, 2026).

- `native_evaluation`: Ultralytics precision, recall, mAP50, mAP50-95, and mAP75, with per-class AP, checkpoint hashes, and evaluation settings. Accuracy values use percentages.
- `coco_size_evaluation`: separately calculated size AP, original-image area ranges, COCO maxDets, and matching ground-truth/checkpoint hashes.
- `reported_complexity`: the manuscript's profiling records, rather than counts inferred from inference-time model fusion.
- `training_settings`: recorded primary n/s settings for each dataset.
- `control_comparisons`: manuscript control tables; `columns` defines the order of each `values` array. The complete-model AP50 references use 48.1%; SIoU remains the independent 47.9% result.

The snapshot includes aggregate metrics and hashes. It contains no private server paths, credentials, dataset images, or trained weights. VisDrone/UAVDT checkpoints were re-evaluated with batch16/FP16; AI-TOD-v2 retains batch15/FP32. COCO size AP uses maxDets=300 and is reported separately from the native Ultralytics metrics.
