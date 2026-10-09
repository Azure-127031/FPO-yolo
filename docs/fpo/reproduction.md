# Reproduction guide

[Project home](../../README.md) · [简体中文](reproduction.zh-CN.md) · [Code design](architecture.md) · [Experiments](experiments.md) · [Development record](development.md)

The released implementation uses Ultralytics 8.3.237. This guide separates entry-point defaults from the dataset-specific settings used for the reported results. Dataset preparation is required; this release does not contain downloaded datasets, annotation-conversion utilities, or trained checkpoints.

## Environment and installation

Use an isolated Python environment with a compatible PyTorch/CUDA installation. The reference combination is Python 3.10, PyTorch 2.4.1, and CUDA 11.8. The VisDrone/UAVDT timing experiments use an NVIDIA L20.

```bash
git clone https://github.com/Azure-127031/FPO-yolo.git FPO-YOLO
cd FPO-YOLO
python -m pip install -e .
python -m pip install -r requirements-fpo.txt
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
python -c "from mmcv.ops import ModulatedDeformConv2d; print('MMCV ops import OK')"
python scripts/check_model.py --scale s
```

`ultralytics.__file__` should resolve inside this checkout. AAD imports MMCV's compiled deformable-convolution operator at module load time. Installing `mmcv-lite`, or installing a wheel compiled for a different PyTorch/CUDA combination, does not supply a compatible runtime. Use [MMCV's official installation/build instructions](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

The requirements file specifies `mmcv>=2.0.0,<2.3.0`; wheel availability depends on the platform and software combination.

## Dataset protocols

| Dataset | Training / validation / final evaluation | Class count | Important distinction |
|---|---|---:|---|
| VisDrone2019-DET | Official train / val / test-dev | 10 | Checkpoint selection uses val, then fixed weights are evaluated on test-dev. |
| UAVDT | 24 / 6 complete sequences drawn from the official training portion; 20,533 / 3,610 images | 3 | Results concern the custom validation split, not the official test set. |
| AI-TOD-v2 | Public train / val / test; 11,214 / 2,804 / 14,018 images | 8 | Train only on train; do not merge train and val. |

Each model is trained separately on its target dataset. The additional datasets examine architecture applicability after retraining, not zero-shot transfer.

### YOLO annotations

An image and its label file share a basename. Each label row is:

```text
class_id x_center y_center width height
```

Class IDs are zero-based and coordinates are normalized by image width/height. Keep annotation and YAML class orders identical. Check that boxes are inside the image and have positive area, retain legitimate images without objects, and preserve the dataset's treatment of excluded/ignored annotations. For VisDrone, ignored regions and non-target categories must not be mapped into the ten target classes.

Use [configs/visdrone.yaml](../../configs/visdrone.yaml) and [configs/uavdt.yaml](../../configs/uavdt.yaml) after replacing their `path`. Their `test` entries point to different subsets: VisDrone uses test-dev, while UAVDT's template leaves room for a separate official test directory. A configured test path does not mean a test result has been reported.

For UAVDT, sort the official training sequence names and select six complete sequences without replacement using seed 42; use the other 24 for training. Preserve sequence membership during conversion and keep the official 20 test sequences separate. Matching aggregate image counts alone is insufficient to reproduce a split: retain the actual sequence lists and data manifests.

Create an AI-TOD-v2 YAML from the **v2 annotations**, preserving the public partition membership and category mapping. Clip converted boxes to image bounds, discard zero-area boxes, and keep empty images. Do not substitute the original AI-TOD annotation version. This release has no automatic AI-TOD converter or dataset template, so prepare and audit this YAML before using the API example.

## Training and evaluation settings

| Setting | VisDrone / UAVDT | AI-TOD-v2 |
|---|---|---|
| Initialization | Model YAML / random initialization | Model YAML / random initialization |
| Image size | 640×640 | 640×640 |
| Batch size | 16 | 15 |
| Maximum epochs | 200 | 200 |
| Early-stopping patience | 100 | 50 |
| Optimizer | SGD | `auto` |
| SGD settings | lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005 | Selected by the released trainer |
| Warm-up | 3 epochs | Trainer defaults |
| Learning-rate schedule | Linear (`cos_lr=False`) | Trainer defaults |
| AMP during training | Enabled | Enabled |
| Mosaic closure | Last 10 epochs | Last 10 epochs |
| Seed | Public training script defaults to 0 | Reported runs use 0 |
| Evaluation precision | FP16 | FP32 |
| Evaluation batch | 16 | 15 |
| Confidence filter | 0.001 in detection evaluator defaults | 0.001 explicitly |
| NMS IoU / max detections | 0.7 / 300 | 0.7 / 300 |
| Test-time augmentation | Disabled | Disabled |

The default seed in `scripts/train.py` is a release setting; it should not be used to infer undocumented seeds for historical runs. Structural comparisons retain the CIoU regression objective. FPO-YOLO uses the inherited classification and distribution-focal-loss training machinery; there is no separate custom training loss introduced by these modules.

### VisDrone and UAVDT entry points

```bash
python scripts/train.py --scale s --data configs/visdrone.yaml --device 0 --project runs/fpo_yolo --name fpo_yolo_s_e200
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split val
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split test --save-json
```

Replace the data argument with `configs/uavdt.yaml` for the UAVDT training/validation setting. Use `--split val` for the UAVDT result table. A repeated run may receive a suffixed output directory; always use its actual checkpoint path.

The training script accepts n/s/m/l/x scales. Reported results cover n/s/m/l on VisDrone and n/s on UAVDT and AI-TOD-v2. Configuration support alone does not establish measured performance for other scales.

### AI-TOD-v2 training and test evaluation

Run the following Python example from the installed checkout **after** creating an audited AI-TOD-v2 YAML. Set `scale` to `n` or `s`. To reproduce the corresponding baseline, use `YOLO(f"yolo11{scale}.yaml")` with the same training/evaluation arguments.

```python
from pathlib import Path
from ultralytics import YOLO

scale = "s"
data = "path/to/audited_aitod_v2.yaml"
model = YOLO(f"yolo11{scale}-fpo.yaml")
model.train(
    data=data,
    imgsz=640,
    epochs=200,
    batch=15,
    optimizer="auto",
    patience=50,
    seed=0,
    amp=True,
    close_mosaic=10,
    device=0,
    project="runs/aitod_v2",
    name=f"fpo_yolo_{scale}",
)

# Use the actual directory, including any suffix added by the trainer.
best = Path(model.trainer.save_dir) / "weights" / "best.pt"
YOLO(str(best)).val(
    data=data,
    split="test",
    imgsz=640,
    batch=15,
    half=False,
    conf=0.001,
    iou=0.7,
    max_det=300,
    augment=False,
    device=0,
    project="runs/aitod_v2_test",
    name=f"fpo_yolo_{scale}",
    save_json=True,
)
```

In this checkout, detection fitness uses validation mAP50–95, as defined in `ultralytics/utils/metrics.py`. This controls `best.pt` selection. The test set must not influence checkpoint or hyperparameter selection. The four reported AI-TOD-v2 runs completed 200 epochs.

The displayed AI-TOD-v2 results use the shared Ultralytics evaluator. They are not directly comparable to a trainval-trained detector evaluated with 1,500 detections per image or an independently implemented tiny-size COCO metric. AP75 and official tiny-size breakdowns are not reported in this release.

## Match the released architecture

- **BCSG:** `C3k2_BCSG` controls bottleneck shortcut behavior, not a C3k branch switch. Its learnable response bank uses contiguous output-channel partitions; do not reinterpret it as an interleaved per-channel derivative bank.
- **AAD:** the implementation calls `ModulatedDeformConv2d` (DCNv2). Its historical `MDCv3` class name is not an operator-version claim.
- **CIHF:** the YAML module name is `ScalSeq_SBA`; this is the released CIHF implementation.
- **PASF:** use `Detect_PASF_Final`, not a historical `Detect_ASFF1`/ASFF head or weights from an unrelated architecture.

Do not silently replace compiled operators or head definitions when comparing models. The class count is taken from the dataset YAML during training. Parameter counts can consequently differ slightly between datasets.

## Record a reproducible run

Keep the code commit, environment versions, dataset/sequence manifests and hashes, actual `args.yaml`, `results.csv`, training log, validation-selected weights, and evaluation outputs. Keep baseline and proposed-model runs in separate directories under an identical per-dataset protocol.

For runtime comparisons, record GPU, precision, batch size, image size, and preprocessing/inference/postprocessing timing. Profiler-estimated GFLOPs may not capture custom operators in the same way as ordinary convolutions; measured latency is a separate quantity. Do not convert a single timing result into a general deployment-speed claim.

## Troubleshooting

| Symptom | First check |
|---|---|
| `No module named mmcv._ext` or missing deformable op | Install an MMCV build with compiled ops matching PyTorch/CUDA; remove conflicting `mmcv-lite` installations from the environment if necessary. |
| The parser does not recognize an FPO module | Inspect `ultralytics.__file__`; the editable checkout may be shadowed by another installation. |
| A checkpoint reports incompatible shapes/classes | Confirm the scale, dataset class order, and final PASF architecture match the checkpoint. |
| Reported test values differ | Check annotation version, partition membership, validation-selected weights, precision, confidence, NMS threshold, max detections, and evaluator implementation. |
| A new run appears under a suffixed directory | Use the trainer's actual `save_dir`; do not evaluate a checkpoint from an older run. |

This guide documents the release and reported protocols. It does not claim that all historical runs can be exactly reconstructed without their original manifests and checkpoints.
