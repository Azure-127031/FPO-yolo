# FPO-YOLO

### A Feature Propagation-Oriented Detector for Small Objects in UAV Imagery

[English](README.md) | [简体中文](README.zh-CN.md) · [Quick start](#quick-start) · [Results](#results) · [Reproduction guide](docs/fpo/reproduction.md) · [Research & figure development](docs/fpo/development.md)

FPO-YOLO is a YOLO11-based aerial detector with **P2–P4 prediction levels** (strides 4, 8, and 16). It connects local feature modulation, adaptive downsampling, cross-level context aggregation, and location-dependent scale selection along a high-resolution detection path.

On **VisDrone2019 validation**, FPO-YOLO-s reaches **47.9% mAP50 / 29.2% mAP50–95**, improving over YOLO11s by **9.1 / 6.0 percentage points** with **56.2% fewer parameters**. The same validation-selected checkpoint reaches **38.9% / 22.8% on test-dev**. The parameter reduction comes with a higher estimated operation count; runtime depends on the dataset and evaluation settings.

<p align="center">
  <img src="docs/fpo/assets/architecture.png" alt="FPO-YOLO architecture: backbone, high-resolution neck, P2–P4 heads, and BCSG container" width="100%">
</p>

The overview preserves the original diagram layout. Scale transitions, shortcut semantics, and feature names are checked against the released configuration. See [how the figures were developed](docs/fpo/development.md#figure-development).

## Updates

- **2026-10-09:** Expanded English/Chinese documentation, added UAVDT and AI-TOD-v2 results with separate evaluation protocols, and documented the figure-development workflow.
- **2026-09-29:** Released the FPO-YOLO implementation based on Ultralytics 8.3.237.
- **2026-09-08:** Created the repository.

## Method

Small objects have limited spatial support. Preserving a shallow prediction level helps, but its features still need contextual information and appropriate scale selection. The four components address different operations on this path:

| Component | Role in the detection path | Implementation |
|---|---|---|
| **BCSG** — bottleneck with channel–spatial gating | Modulates local features with cascaded residual spatial and channel gates; the response bank is initialized with Scharr and mean kernels and remains learnable. | [fpo_bcsg.py](ultralytics/nn/modules/fpo_bcsg.py) |
| **AAD** — adaptive aggregation downsampling | Aggregates gated deformable responses before reducing resolution, alongside a regular downsampling branch. | [fpo_aad.py](ultralytics/nn/modules/fpo_aad.py) |
| **CIHF** — cross-level interaction and high-resolution fusion | Updates middle/deep context through conditional interaction, then aligns and aggregates it onto a shallow residual representation. | [fpo_cihf.py](ultralytics/nn/modules/fpo_cihf.py) |
| **PASF** — position-adaptive scale fusion | Aligns neck features and selects their contributions separately at each output location before prediction. | [fpo_pasf.py](ultralytics/nn/modules/fpo_pasf.py) |

The components are evaluated in a shared high-resolution configuration. Combination and removal ablations support their complementary roles within this path. [Ablation results and design lineage →](docs/fpo/development.md)

## Results

All accuracy values below are percentages. mAP50–95 averages AP over IoU thresholds 0.50:0.05:0.95. **Validation and test results are listed separately.**

### VisDrone2019-DET

| Model | Val mAP50 | Val mAP50–95 | Test-dev mAP50 | Test-dev mAP50–95 | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s | 38.8 | 23.2 | 31.4 | 18.0 | 9.43 | 21.6 |
| FPO-YOLO-n | 40.9 | 24.7 | 33.1 | 19.2 | 1.34 | 13.5 |
| FPO-YOLO-s | 47.9 | 29.2 | 38.9 | 22.8 | 4.13 | 36.5 |
| FPO-YOLO-m | 51.5 | 31.8 | 41.1 | 24.3 | 14.84 | 121.0 |
| FPO-YOLO-l | 52.2 | 32.4 | 41.4 | 24.5 | 16.50 | 138.0 |

The official train/val/test-dev partitions are retained. Test-dev is evaluated using validation-selected weights. GFLOPs are profiler estimates at a 640-pixel input, not a hardware-independent latency measure. The configuration also supports scale `x`; no `x` result is reported here.

### UAVDT — sequence-level validation

| Model | Val mAP50 | Val mAP50–95 |
|---|---:|---:|
| YOLO11n | 37.0 | 18.7 |
| FPO-YOLO-n | 44.0 | 23.5 |
| YOLO11s | 44.4 | 24.9 |
| FPO-YOLO-s | 52.0 | 28.3 |

The custom split uses **24 complete training sequences / 6 validation sequences**, containing **20,533 / 3,610 images**. Validation sequences are sampled from the sorted official training sequence list with seed 42. The official test portion is excluded from this split; these values are **validation results**. Aggregate gains coexist with class-level localization trade-offs and increased measured latency on this dataset.

### AI-TOD-v2 — official test split

| Model | Test mAP50 | Test mAP50–95 |
|---|---:|---:|
| YOLO11n | 34.4 | 14.8 |
| FPO-YOLO-n | 43.1 | 19.5 |
| YOLO11s | 45.5 | 19.9 |
| FPO-YOLO-s | 54.4 | 25.0 |

The public **11,214 / 2,804 / 14,018** train/val/test split is retained. Models are trained only on `train`, selected on `val`, and evaluated on `test` with FP32 and `max_det=300`. These are results from the shared Ultralytics evaluator; they are not a reimplementation of the official tiny-size COCO evaluation. UAVDT and AI-TOD-v2 use **separately trained models**, rather than zero-shot transfer of VisDrone weights. [Dataset-specific protocols →](docs/fpo/reproduction.md#dataset-protocols)

## Qualitative comparison

<p align="center">
  <img src="docs/fpo/assets/qualitative.png" alt="YOLO11s and FPO-YOLO-s detections and detection-conditioned Grad-CAM on four VisDrone scenes" width="100%">
</p>

The four rows compare detections and detection-conditioned Grad-CAM in vehicle-centered regions. The paired CAMs use the same ground-truth vehicle group within each region. They illustrate prediction differences; they are not a quantitative explanation of all component effects. The image panels originate from VisDrone, with detector outputs and visualization added by this project. [Visualization procedure →](docs/fpo/development.md#detection-and-response-visualization)

## Quick start

### 1. Install the repository and compiled operators

The reference software environment is **Python 3.10, PyTorch 2.4.1, CUDA 11.8, and Ultralytics 8.3.237**. Install a PyTorch/CUDA combination suitable for your machine first, then use an isolated environment:

```bash
git clone https://github.com/Azure-127031/FPO-yolo.git FPO-YOLO
cd FPO-YOLO
python -m pip install -e .
python -m pip install -r requirements-fpo.txt
```

AAD requires **MMCV with compiled ops**, including `mmcv.ops.ModulatedDeformConv2d`. `mmcv-lite` is insufficient. Match the MMCV build to PyTorch and CUDA; if no compatible wheel is available, follow the [official MMCV installation instructions](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). Confirm that the editable installation resolves to this checkout:

```bash
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
python -c "from mmcv.ops import ModulatedDeformConv2d; print('MMCV ops import OK')"
python scripts/check_model.py --scale s
```

The import check verifies operator availability, while `check_model.py` builds the architecture. Neither substitutes for a successful CUDA training run.

### 2. Prepare YOLO-format data

Set `path` in [configs/visdrone.yaml](configs/visdrone.yaml) to your dataset root:

```text
VisDrone2019/
├── images/
│   ├── train/
│   ├── val/
│   └── test-dev/
└── labels/
    ├── train/
    ├── val/
    └── test-dev/
```

Each image has a same-stem label file with rows `class_id x_center y_center width height`, normalized to `[0, 1]`. Retain the class order in the YAML. Obtain data from the dataset authors; full datasets and conversion tools are not distributed in this release. [Split and annotation requirements →](docs/fpo/reproduction.md#dataset-protocols)

### 3. Train from random initialization

```bash
python scripts/train.py --scale s --data configs/visdrone.yaml --device 0
```

The released training entry point uses YAML initialization, `imgsz=640`, batch 16, up to 200 epochs, SGD, patience 100, AMP, and `close_mosaic=10`. Its default seed is 0. This is the release default, not a retrospective claim that every historical experiment used that seed. The script is intended for the VisDrone/UAVDT settings; **AI-TOD-v2 requires the separate protocol below**.

### 4. Evaluate validation-selected weights

```bash
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split val
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split test --save-json
```

For this YAML, `--split test` maps to `images/test-dev`. Use the actual run directory if Ultralytics adds a suffix. Freeze checkpoint selection before test evaluation; do not substitute `last.pt` for `best.pt`.

`scripts/val.py` uses FP16 and batch 16. Use the [AI-TOD-v2 API example](docs/fpo/reproduction.md#ai-tod-v2-training-and-test-evaluation) for its FP32/batch-15 protocol.

## Research and figure development

The project moved from a high-resolution prediction path to module comparisons, multi-dataset evaluation, and implementation-aligned figures. The development record explains:

- why local gating, sampling, context aggregation, and output-side selection are treated as complementary operations;
- how the original PowerPoint diagrams were checked against code and refined without changing their geometry;
- how module fills were recolored individually, rather than applying a page-wide tint;
- how plots, subfigure labels, detection panels, and Grad-CAM were prepared for the manuscript.

**[Read the development record (English)](docs/fpo/development.md) · [阅读研究与图件演进记录（中文）](docs/fpo/development.zh-CN.md)**

## Repository guide

| Path | Contents |
|---|---|
| [ultralytics/cfg/models/11/yolo11-fpo.yaml](ultralytics/cfg/models/11/yolo11-fpo.yaml) | Shared architecture and n/s/m/l/x scale definitions |
| [ultralytics/nn/modules](ultralytics/nn/modules) | FPO modules and inherited Ultralytics components |
| [scripts](scripts) | Architecture inspection, training, and evaluation entry points |
| [configs](configs) | VisDrone and UAVDT dataset templates |
| [docs/fpo](docs/fpo) | Reproduction guide, development record, and figure assets |

## Availability and scope

This release includes implementation, architecture/dataset configurations, entry-point scripts, and documentation. **Trained checkpoints and full raw experiment logs are not included.** The displayed figures are exported research assets; they are not a complete package for recomputing every visualization from weights.

Custom deformable operators need a compatible compiled environment. ONNX/TensorRT export, deployment speed, and support for arbitrary package versions are not established by this release. FPO-YOLO is parameter-efficient, but it does not reduce GFLOPs at every scale or improve every class metric.

## Acknowledgements, citation, and license

The code builds on [Ultralytics](https://github.com/ultralytics/ultralytics) and [MMCV](https://github.com/open-mmlab/mmcv). AAD uses modified DCNv2 sampling; CIHF builds on SBA-style interaction; PASF uses ASFF-style spatial scale weighting and coordinate-attention-style directional refinement. The [development record](docs/fpo/development.md#design-lineage) distinguishes these antecedents from the modifications implemented here.

The associated manuscript is **FPO-YOLO: A Feature Propagation-Oriented Detector for Small Objects in UAV Imagery**. Publication metadata is not provided in this release. Until a formal citation is available, reference this repository together with the commit used in your experiment.

This repository is distributed under the [AGPL-3.0 license](LICENSE). Please retain upstream notices. For issues, include the commit, environment versions, dataset YAML, command, and full traceback; do not attach private data or credentials.
