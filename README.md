# FPO-YOLO

### A Feature Propagation-Oriented Detector for Small Objects in UAV Imagery

[English](README.md) | [简体中文](README.zh-CN.md) · [Quick start](#quick-start) · [Results](#results) · [Code design](docs/fpo/architecture.md) · [Experiments](docs/fpo/experiments.md) · [Research & figures](docs/fpo/development.md)

FPO-YOLO is a YOLO11-based aerial detector with **P2–P4 prediction levels** (strides 4, 8, and 16). It connects local feature modulation, adaptive downsampling, cross-level context aggregation, and location-dependent scale selection along a high-resolution detection path.

On **VisDrone2019 validation**, FPO-YOLO-s reaches **48.1% mAP50 / 29.2% mAP50–95**, improving over YOLO11s by **9.3 / 6.1 percentage points** with **56.2% fewer parameters**. The same validation-selected checkpoint reaches **38.9% / 22.7% on test-dev**. The parameter reduction comes with a higher estimated operation count; runtime depends on the dataset and evaluation settings.

<p align="center">
  <img src="docs/fpo/assets/architecture.png" alt="FPO-YOLO architecture: backbone, high-resolution neck, P2–P4 heads, and BCSG container" width="100%">
</p>

**Architecture.** BCSG and AAD process local features and scale transitions; CIHF supplies context to the high-resolution neck; PASF fuses the three neck outputs separately for P2, P3, and P4 prediction. [Layer-by-layer code guide →](docs/fpo/architecture.md)

## Reading guide

| What would you like to understand? | Start here |
|---|---|
| The method and final results | [Method](#method) and [results](#results) on this page |
| What changes relative to YOLO11, and how the code connects | [Code design](docs/fpo/architecture.md): YAML layers, four component computations, and integration |
| Why each experiment was conducted and what its results support | [Experiment history](docs/fpo/experiments.md): E1–E8 comparisons, training/resume, fixed testing, and dated milestones |
| How to install, train, and evaluate | [Quick start](#quick-start) → [full reproduction guide](docs/fpo/reproduction.md) |
| How the project developed from early drawings to the paper and second release | [Research and figure history](docs/fpo/development.md): timeline, tools, revisions, and before/after figures |

## Updates

- **2026-10-11:** Synchronized validation/test-dev results and complete-model references with the current manuscript; added AP75, COCO size AP, updated plots, and a [machine-readable result snapshot](docs/fpo/data/reported-results.json).

- **2026-10-10:** Refined bilingual navigation and usage examples, added scale-matched test-dev comparisons and a software citation, and retained the full code, experiment, and figure-development records.
- **2026-10-09:** Expanded second-edition English/Chinese documentation with UAVDT and AI-TOD-v2 results and their protocols; added code-design explanations, experiment history, earlier milestones, and the figure-development workflow.
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

BCSG changes the feature responses available to later stages, while AAD changes where those responses are sampled during resolution reduction. CIHF provides image- and channel-dependent context weights to the shallow residual path. PASF then selects scales at individual prediction locations, where CIHF's spatially shared weights cannot make separate decisions. These computations connect feature formation, scale transition, contextual aggregation, and prediction.

**Combination evidence.** Relative to the shared CIHF reference, AAD, BCSG, and PASF individually add 0.4, 1.0, and 0.7 points in validation mAP50–95. Their full combination adds 2.6 points, exceeding the 2.1-point sum; removing one component from the full model costs 0.6–0.9 points. These comparisons support complementarity within the evaluated configuration. [Implementation walkthrough →](docs/fpo/architecture.md) · [Combination and removal results →](docs/fpo/experiments.md#e2-component-combinations-and-removal) · [Design lineage →](docs/fpo/development.md#design-lineage)

## Results

Accuracy values are percentages, rounded to one decimal place; reported gains use the unrounded results. mAP50–95 averages AP over IoU thresholds 0.50:0.05:0.95. **Validation and test results are listed separately.**

### VisDrone2019-DET

| Model | Val mAP50 | Val mAP50–95 | Test-dev mAP50 | Test-dev mAP50–95 | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 32.3 | 18.7 | 26.0 | 14.6 | 2.58 | 6.3 |
| FPO-YOLO-n | 41.2 | 24.6 | 33.0 | 19.1 | 1.34 | 13.5 |
| YOLO11s | 38.8 | 23.2 | 31.7 | 18.3 | 9.43 | 21.6 |
| FPO-YOLO-s | 48.1 | 29.2 | 38.9 | 22.7 | 4.13 | 36.5 |
| FPO-YOLO-m | 51.5 | 31.8 | 41.1 | 24.3 | 14.84 | 121.0 |
| FPO-YOLO-l | 52.2 | 32.4 | 41.4 | 24.5 | 16.50 | 138.0 |

The official train/val/test-dev partitions are retained. Test-dev is evaluated using validation-selected weights. GFLOPs are profiler estimates at a 640-pixel input, not a hardware-independent latency measure. The configuration also supports scale `x`; no `x` result is reported here.

<details>
<summary><strong>Scale-matched YOLO11 comparison on test-dev</strong></summary>

| Model | Test-dev mAP50 | Test-dev mAP50–95 | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|
| YOLO11n | 26.0 | 14.6 | 2.58 | 6.3 |
| FPO-YOLO-n | **33.0** | **19.1** | **1.34** | 13.5 |
| YOLO11s | 31.7 | 18.3 | 9.43 | 21.6 |
| FPO-YOLO-s | **38.9** | **22.7** | **4.13** | 36.5 |
| YOLO11m | 35.3 | 20.8 | 20.04 | 67.7 |
| FPO-YOLO-m | **41.1** | **24.3** | **14.84** | 121.0 |
| YOLO11l | 36.4 | 21.7 | 25.32 | 87.3 |
| FPO-YOLO-l | **41.4** | **24.5** | **16.50** | 138.0 |

Bold marks higher accuracy or fewer parameters within each scale. The improvements come with higher estimated operation counts. Checkpoints are selected on validation data before test-dev evaluation.

</details>

### UAVDT — sequence-level validation

| Model | Val mAP50 | Val mAP50–95 |
|---|---:|---:|
| YOLO11n | 37.0 | 18.7 |
| FPO-YOLO-n | 44.1 | 23.4 |
| YOLO11s | 44.4 | 24.9 |
| FPO-YOLO-s | 52.0 | 28.3 |

The custom split uses **24 complete training sequences / 6 validation sequences**, containing **20,533 / 3,610 images**. Validation sequences are sampled from the sorted official training sequence list with seed 42. The official test portion is excluded from this split; these values are **validation results**. Aggregate gains coexist with car/truck AP trade-offs and increased measured latency on this dataset.

### AI-TOD-v2 — official test split

| Model | Test mAP50 | Test mAP50–95 |
|---|---:|---:|
| YOLO11n | 34.4 | 14.8 |
| FPO-YOLO-n | 43.1 | 19.5 |
| YOLO11s | 45.5 | 19.9 |
| FPO-YOLO-s | 54.4 | 25.0 |

The public **11,214 / 2,804 / 14,018** train/val/test split is retained. Models are trained only on `train`, selected on `val`, and evaluated on `test` with FP32 and `max_det=300`. Overall accuracy and AP75 use Ultralytics; COCO size AP is reported separately below with the same 300-detection limit. UAVDT and AI-TOD-v2 use **separately trained models**, rather than zero-shot transfer of VisDrone weights. [Dataset-specific protocols →](docs/fpo/reproduction.md#dataset-protocols)

### AP75 and object-size results

The same selected checkpoints supply these strict-IoU results. Full precision values, per-class AP, weight hashes, and evaluation settings are in the [result JSON](docs/fpo/data/reported-results.json).

| Model | VisDrone val mAP75 | UAVDT val mAP75 | AI-TOD-v2 test mAP75 |
|---|---:|---:|---:|
| YOLO11n | 18.6 | 14.7 | 10.3 |
| FPO-YOLO-n | 25.1 | 21.4 | 14.5 |
| YOLO11s | 23.6 | 24.2 | 14.3 |
| FPO-YOLO-s | 30.2 | 25.7 | 19.2 |

<details>
<summary><strong>COCO size-specific AP</strong></summary>

**VisDrone2019 validation**

| Model | AP_small | AP_medium | AP_large |
|---|---:|---:|---:|
| YOLO11n | 8.7 | 26.9 | 35.2 |
| FPO-YOLO-n | 14.9 | 33.3 | 43.6 |
| YOLO11s | 12.4 | 32.9 | 42.4 |
| FPO-YOLO-s | 18.8 | 38.7 | 44.4 |

**UAVDT validation**

| Model | AP_small | AP_medium | AP_large |
|---|---:|---:|---:|
| YOLO11n | 10.5 | 26.1 | 38.0 |
| FPO-YOLO-n | 12.5 | 28.0 | 27.9 |
| YOLO11s | 10.1 | 25.2 | 37.9 |
| FPO-YOLO-s | 12.7 | 39.7 | 35.6 |

**AI-TOD-v2 test**

| Model | AP_verytiny | AP_tiny | AP_small | AP_medium |
|---|---:|---:|---:|---:|
| YOLO11n | 3.5 | 12.7 | 20.9 | 28.2 |
| FPO-YOLO-n | 7.1 | 18.0 | 25.2 | 31.6 |
| YOLO11s | 4.7 | 17.1 | 26.8 | 37.0 |
| FPO-YOLO-s | 10.3 | 23.2 | 31.7 | 38.8 |

Size AP uses original-image area, COCOeval, and maxDets=300. VisDrone/UAVDT boundaries are 32²/96²; AI-TOD-v2 boundaries are 8²/16²/32². These values are reported separately from Ultralytics AP.

</details>

### Controlled design comparisons

The [E1–E8 experiment index](docs/fpo/experiments.md) connects each design decision to its comparison: prediction levels and CIHF; component combinations and removal; sampling variants; BCSG response organization; fusion heads; regression losses; model scales; and separately trained additional datasets. It distinguishes structural gains from component gains and reports the associated resource costs.

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

These checks confirm that Python resolves this checkout, the compiled operator can be imported, and the selected architecture can be constructed.

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
python scripts/train.py --scale s --data configs/visdrone.yaml --batch 12 --device 0
```

The training entry point uses YAML initialization, `imgsz=640`, up to 200 epochs, SGD, patience 100, AMP, and `close_mosaic=10`, with seed 0. The reported VisDrone FPO-n/s runs use `--batch 12`; UAVDT uses `--batch 16`. The command above matches the VisDrone setting. The script retains 16 as its default and accepts an explicit batch override. The script is intended for the VisDrone/UAVDT settings; **AI-TOD-v2 requires the separate protocol below**.

### 4. Evaluate validation-selected weights

```bash
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split val
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split test --save-json
```

For this YAML, `--split test` maps to `images/test-dev`. Use the actual run directory if Ultralytics adds a suffix. Freeze checkpoint selection before test evaluation; do not substitute `last.pt` for `best.pt`.

`scripts/val.py` uses FP16 and batch 16. Use the [AI-TOD-v2 API example](docs/fpo/reproduction.md#ai-tod-v2-training-and-test-evaluation) for its FP32/batch-15 protocol.

### 5. Predict with your trained checkpoint

```python
from ultralytics import YOLO

model = YOLO("runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt")
model.predict(source="path/to/image.jpg", imgsz=640, conf=0.25, device=0, save=True)
```

Replace the paths with your checkpoint and input. Here `conf=0.25` is a display filter for predictions; quantitative evaluation uses the settings in the [reproduction guide](docs/fpo/reproduction.md), including a 0.001 confidence filter. This release does not provide pretrained checkpoints.

## Research and figure development

From the early component drawings to the second public edition, the work progressed through **module illustration, detection-path organization, code/result checks, additional-dataset evaluation, and manuscript figure refinement**. The drawings developed alongside the research: each figure needed to explain the data flow and match the implementation.

<details>
<summary><strong>Research milestones: early drawings to the second release</strong></summary>

| Period | Main development |
|---|---|
| March–July 2026¹ | Retained YOLO11 baseline, BCSG, AAD, and SBA/ASFF drawings; iterated PowerPoint sources and prepared VisDrone dataset/scene illustrations. |
| August 2026¹ | Organized PPTX/PDF versions of the FPO overview, baseline, and component diagrams. |
| September 3–16, 2026 | Brought together the manuscript, code, and logs; revised methods, ablations, class results, and figures; added the UAVDT section. Created the repository on September 8. |
| September 18–29, 2026 | Prepared the English double-column manuscript and protocol notes, aligned the three BCSG panels, and released the implementation on September 29. |
| October 1–6, 2026 | Reviewed supplementary protocols and completed AI-TOD-v2 auditing, four training runs, and fixed-test evaluation. |
| October 7–8, 2026 | Incorporated results and peer feedback, checked diagram semantics, preserved original shapes/routes, and refined module colors and experimental plots. |
| October 9, 2026 | Published bilingual second-edition documentation, added manuscript repository links, and connected the earlier research and figure records. |

¹ Early months reflect retained figure-file timestamps, not inferred experiment-completion dates. The full timeline identifies each date source.

</details>

The figure workflow combines **PowerPoint → PDF** architecture exports, **LaTeX/TikZ** assembly of BCSG panels, **Matplotlib** quantitative plots, and panels from predictions and detection-conditioned Grad-CAM. The detailed record includes before/after colors and explains the problems addressed in each revision.

**[Read the full research and drawing history →](docs/fpo/development.md)** · [Figure methods](docs/fpo/development.md#figure-development) · [简体中文](docs/fpo/development.zh-CN.md)

## Repository guide

| Path | Contents |
|---|---|
| [ultralytics/cfg/models/11/yolo11-fpo.yaml](ultralytics/cfg/models/11/yolo11-fpo.yaml) | Shared architecture and n/s/m/l/x scale definitions |
| [ultralytics/nn/modules](ultralytics/nn/modules) | FPO modules and inherited Ultralytics components |
| [scripts](scripts) | Architecture inspection, training, and evaluation entry points |
| [configs](configs) | VisDrone and UAVDT dataset templates |
| [docs/fpo](docs/fpo) | Code design, experiment history, reproduction, research/figure history, and figure assets |

## Availability and scope

This release includes implementation, architecture/dataset configurations, entry-point scripts, and documentation. **Trained checkpoints and full raw experiment logs are not included.** The displayed figures are exported research assets; they are not a complete package for recomputing every visualization from weights.

Custom deformable operators need a compatible compiled environment. ONNX/TensorRT export, deployment speed, and support for arbitrary package versions are not established by this release. FPO-YOLO is parameter-efficient, but it does not reduce GFLOPs at every scale or improve every class metric.

## Acknowledgements, citation, and license

The code builds on [Ultralytics](https://github.com/ultralytics/ultralytics) and [MMCV](https://github.com/open-mmlab/mmcv). AAD uses modified DCNv2 sampling; CIHF builds on SBA-style interaction; PASF uses ASFF-style spatial scale weighting and coordinate-attention-style directional refinement. The [development record](docs/fpo/development.md#design-lineage) distinguishes these antecedents from the modifications implemented here.

The associated manuscript is **FPO-YOLO: A Feature Propagation-Oriented Detector for Small Objects in UAV Imagery**. Publication metadata is not provided in this release. Until a formal manuscript citation is available, use the software reference below and record the commit used in your experiment. This entry cites the repository, not a published article.

```bibtex
@misc{fpoyolo2026code,
  title = {FPO-YOLO: Open-source implementation},
  year = {2026},
  howpublished = {\url{https://github.com/Azure-127031/FPO-yolo}}
}
```

This repository is distributed under the [AGPL-3.0 license](LICENSE). Please retain upstream notices. For issues, include the commit, environment versions, dataset YAML, command, and full traceback; do not attach private data or credentials.
