# Experiment history and evidence index

[Project home](../../README.md) · [简体中文](experiments.zh-CN.md) · [Code design](architecture.md) · [Research and figures](development.md) · [Reproduction](reproduction.md)

The experiments address three questions: **Does the high-resolution detection path help? Do its components make distinguishable contributions? Do separately trained models also improve on additional datasets?** This page connects the existing manuscript tables, retained experiment records, and second-edition results to the decisions they support.

No models were retrained for this documentation update. E1–E8 are reading indices, not the order in which runs were launched; individually verified run dates are not provided for every VisDrone comparison. Accuracy is reported in percent and differences in percentage points. Full raw logs and trained checkpoints are not released, so this page is a results/protocol index rather than a complete run archive.

## Experiment route

| Index | Question | Comparison |
|---|---|---|
| [E1](#e1-resolution-and-fusion) | Is cross-level interaction useful after introducing a finer grid? | Baseline → P2–P4 → aggregation without interaction → CIHF |
| [E2](#e2-component-combinations-and-removal) | Does each additional component contribute? | Single components, pairs, and the full combination on a shared CIHF reference |
| [E3](#e3-aad-sampling-variants) | Which dynamic sampling implementation is used in AAD? | DCN replacements and modified DCNv2 |
| [E4](#e4-bcsg-response-organization) | Does response grouping affect BCSG? | Per-channel pairing versus contiguous grouping |
| [E5](#e5-head-replacements) | Which prediction-end fusion is selected? | Head replacements within the same framework |
| [E6](#e6-regression-losses) | Does the design depend on replacing the regression loss? | CIoU and alternative losses |
| [E7](#e7-model-scales-and-held-out-evaluation) | Does the final design extend across scales and to held-out evaluation? | n/s/m/l and VisDrone test-dev |
| [E8](#e8-additional-dataset-experiments) | Do separately trained models improve on other datasets? | UAVDT sequence validation and AI-TOD-v2 fixed testing |

## Dated experiment milestones

| Date | Recorded work | Consequence |
|---|---|---|
| September 4–6, 2026 | Revised the experiment chapter against validation records, organizing structural comparisons, combinations, and class results. | Connected component claims to comparison tables. These are revision dates, not completion dates for individual runs. |
| September 9–10, 2026 | Continued method/code/literature checks and added the UAVDT section. | Extended evaluation beyond the main dataset. |
| September 11–19, 2026 | Updated validation/test-dev tables, checked reporting consistency, and recorded UAVDT protocol questions separately. | Clarified splits, metrics, and complexity measurements instead of carrying forward inconsistent draft numbers. |
| September 29, 2026 | Published implementation [`9016224`](https://github.com/Azure-127031/FPO-yolo/commit/9016224). | Released final modules, configurations, and entry points. |
| October 1–2, 2026 | Reviewed UAVDT sequence coverage and internal validation versus official testing. | Identified the reported split as sequence-level validation, not an official test result. |
| October 3, 2026 | Prepared AI-TOD-v2, audited 56 Parquet shards and annotation conversion, checked fixed splits and code snapshots, and retained baseline training records. | Matched data audits and code snapshots across two training machines. |
| October 5, 2026 | Resumed FPO-YOLO-s from saved training state and started FPO-YOLO-n from random initialization. | Preserved resume provenance and checked scales and settings. |
| October 6, 2026 | Confirmed all four YOLO11n/s and FPO-YOLO-n/s runs completed 200 epochs; evaluated validation-selected checkpoints on the fixed test set. | Produced a common test summary without selecting checkpoints on test results. |
| October 7–9, 2026 | Integrated supplementary results into the manuscript, checked figures/text/protocols, and published second-edition documentation. | Connected results to code, protocols, and development history. |

Earlier figures and manuscript milestones appear in the [full research timeline](development.md#documented-milestones). Figure-file modification times identify retained versions, not experiment dates.

## E1 Resolution and fusion

These VisDrone validation comparisons distinguish a finer prediction grid from additional cross-level context.

| Configuration | mAP50 | mAP50–95 |
|---|---:|---:|
| YOLO11s | 38.8 | 23.2 |
| P2–P4 prediction levels | 42.8 | 25.9 |
| Shallow aggregation without cross-level interaction | 43.1 | 26.1 |
| Full CIHF reference | 43.9 | 26.6 |

P2–P4 adds 2.7 mAP50–95 points over the baseline; full CIHF adds 0.5 over aggregation without interaction. CIHF then serves as the shared reference for local representation, downsampling, and prediction-end fusion comparisons. [Configuration walkthrough](architecture.md#how-the-configuration-connects)

## E2 Component combinations and removal

A = AAD, B = BCSG, and P = PASF. Every row includes CIHF and uses the common VisDrone experimental setting.

| Components added to CIHF | mAP50 | mAP50–95 |
|---|---:|---:|
| None | 43.9 | 26.6 |
| A | 44.6 | 27.0 |
| B | 45.6 | 27.6 |
| P | 45.0 | 27.3 |
| A + B | 46.5 | 28.3 |
| B + P | 46.6 | 28.6 |
| A + P | 46.4 | 28.3 |
| A + B + P | 48.1 | 29.2 |

The individual A/B/P increments are 0.4/1.0/0.7 mAP50–95 points. The full combination gains 2.6 over CIHF, exceeding the sum of those isolated increments by 0.5. Removing A/B/P from the full model reduces mAP50–95 by 0.6/0.9/0.9 points, respectively.

These comparisons support contributions and combined effects under this configuration. Increments depend on the reference; they do not establish positive interaction for every insertion order or statistical significance across multiple seeds.

## E3 AAD sampling variants

Dynamic operators are replaced at the AAD sampling position within the common experimental framework.

| Sampling variant | mAP50 | mAP50–95 |
|---|---:|---:|
| DCNv1 | 47.0 | 28.8 |
| DCNv2 | 46.7 | 28.7 |
| DCNv3 | 46.3 | 28.3 |
| DCNv4 | 47.1 | 28.7 |
| Modified DCNv2, final AAD | 48.1 | 29.2 |

The final implementation compresses offsets, applies a learnable amplitude and global gate, and aggregates dynamic responses before reducing resolution. These are replacement results in this framework, not complete model scores from the original DCN papers. The historical class name `MDCv3` currently calls MMCV `ModulatedDeformConv2d`. [AAD implementation](architecture.md#aad-aggregates-deformable-responses-before-downsampling)

## E4 BCSG response organization

| Response organization | Precision | Recall | mAP50 | mAP50–95 | Parameters (M) |
|---|---:|---:|---:|---:|---:|
| Per-channel pairing | 58.8 | 44.2 | 47.2 | 28.7 | 3.99 |
| Contiguous grouping, current version | 57.9 | 45.3 | 48.1 | 29.2 | 4.13 |

The selected configuration has higher AP and recall, more parameters, and lower precision. It therefore does not improve every metric. This comparison also does not isolate each historical convolution or residual-design change in BCSG. [BCSG implementation and lineage](architecture.md#bcsg-from-multiplicative-gating-to-residual-modulation)

## E5 Head replacements

| Replacement within the common framework | mAP50 | mAP50–95 |
|---|---:|---:|
| Adapted ASFF | 47.1 | 28.8 |
| AFPN3 | 47.0 | 29.0 |
| MBConv | 45.0 | 27.2 |
| LQE | 44.7 | 27.1 |
| DyHead | 44.6 | 26.6 |
| PASF, final version | 48.1 | 29.2 |

PASF achieves the highest AP in this replacement set. Its complete detector has 4.13 M parameters and 36.5 GFLOPs, compared with 5.11 M and 41.2 GFLOPs for adapted ASFF. These counts describe the entire detector. The comparison supports PASF as a whole; it does not separately isolate temperature, equal-weight initialization, or directional attention. [Final head implementation](architecture.md#pasf-from-spatial-scale-weighting-to-an-isolated-final-head)

## E6 Regression losses

| Regression loss | mAP50 | mAP50–95 |
|---|---:|---:|
| DIoU | 47.4 | 29.0 |
| GIoU | 47.3 | 28.9 |
| EIoU | 46.9 | 28.9 |
| SIoU | 47.9 | 29.3 |
| PIoU | 46.7 | 28.4 |
| Shape-IoU | 45.5 | 26.4 |
| Inner-CIoU | 46.5 | 28.3 |
| CIoU, retained | 48.1 | 29.2 |

SIoU is 0.1 mAP50–95 points higher in this comparison. CIoU is retained to keep a consistent regression framework for the main structural comparison. CIoU has the highest mAP50; SIoU remains close in averaged AP.

## E7 Model scales and held-out evaluation

The [homepage VisDrone table](../../README.md#visdrone2019-det) lists the complete n/s/m/l results. For the s scale:

| Check | YOLO11s | FPO-YOLO-s | Difference |
|---|---:|---:|---|
| Val mAP50 / mAP50–95 | 38.8 / 23.2 | 48.1 / 29.2 | +9.3 / +6.1 points |
| Test-dev mAP50 / mAP50–95 | 31.7 / 18.3 | 38.9 / 22.7 | +7.1 / +4.5 points |
| Parameters (M) | 9.43 | 4.13 | 56.2% fewer |
| GFLOPs | 21.6 | 36.5 | Increased operation count |

Test-dev uses the same validation-selected weights. Parameter savings coexist with increased computation; runtime for the P2 path depends on hardware, precision, and data distribution. Predictions and detection-conditioned Grad-CAM provide supplementary observations, not quantitative component ablations. [Figure and visualization workflow](development.md#figure-development)

## E8 Additional-dataset experiments

### UAVDT: establish the split before interpreting results

Complete validation sequences are selected from the official training portion using sorted sequence names and seed 42: 24/6 train/val sequences and 20,533/3,610 images. FPO-n/s achieve 44.1/23.4 and 52.0/28.3 validation mAP50/mAP50–95, versus 37.0/18.7 and 44.4/24.9 for their baselines.

Protocol reviews from September to early October clarified sequence coverage and metrics. These internal validation results are not mixed directly with official-test scores from the literature. Overall AP gains also coexist with class-specific localization tradeoffs and increased measured latency.

### AI-TOD-v2: acquisition, auditing, training, resume, and fixed testing

1. **Prepare data.** On October 3, audit all 56 shards while retaining the 11,214/2,804/14,018 train/val/test split. The raw annotation audit covers 28,036 images and 752,746 boxes, recording 51 zero-area boxes and 28 images with empty annotations. Conversion handles invalid boxes and out-of-bounds coordinates while retaining splits and recording the changes.
2. **Check provenance.** Match code snapshots and data audits across two training machines. Baseline and FPO models are independently trained on this dataset from random initialization, rather than transferred from VisDrone for zero-shot testing.
3. **Preserve resume provenance.** On October 5, resume FPO-s from saved training state and start FPO-n. Final evaluation still uses validation-selected `best.pt`.
4. **Freeze selection before testing.** On October 6, confirm all four runs completed 200 epochs. The recorded best epoch is 200 for all four; this does not generally make `last.pt` interchangeable with `best.pt`. Use the same evaluator on the fixed test set.
5. **Repair result saving.** An issue in result saving after the first baseline inference is fixed in the saving wrapper; export is repeated with identical weights, data, and evaluation settings. It does not lead to model tuning or checkpoint selection on test scores.

Test settings are 640 input, FP32, batch 15, `conf=0.001`, NMS IoU 0.7, `max_det=300`, rectangular inference, and no TTA. Training follows a separate AI-TOD-v2 protocol, including `optimizer=auto` and patience 50; the VisDrone entry point's default SGD settings should not be substituted. See the [reproduction API example](reproduction.md#ai-tod-v2-training-and-test-evaluation).

| Model | Test precision | Test recall | Test mAP50 | Test mAP50–95 |
|---|---:|---:|---:|---:|
| YOLO11n | 59.79 | 33.71 | 34.38 | 14.83 |
| FPO-YOLO-n | 66.65 | 42.63 | 43.10 | 19.46 |
| YOLO11s | 57.10 | 45.38 | 45.53 | 19.87 |
| FPO-YOLO-s | 60.53 | 56.33 | 54.40 | 24.95 |

The homepage rounds these same results to one decimal place. This table uses the Ultralytics evaluator. The new COCO size AP breakdown is reported separately and uses maxDets=300.

## AP75 and size analysis

The October 11 checkpoint evaluation adds strict-IoU AP and original-coordinate COCO size AP for VisDrone, UAVDT, and AI-TOD-v2. See the [homepage tables](../../README.md#ap75-and-object-size-results) and [full numeric snapshot](data/reported-results.json). Full-model AP50 references in E2–E6 use 48.1%; other control measurements retain their recorded values, including SIoU 47.9%. VisDrone/UAVDT primary evaluation uses batch 16 / FP16, and AI-TOD-v2 retains batch 15 / FP32.

## Reading and verification

- **Final public results:** [Project home](../../README.md#results).
- **How the modifications execute:** [Code design](architecture.md), alongside the YAML and each module's `forward`.
- **Environment and dataset preparation:** [Reproduction](reproduction.md), checking each dataset's protocol.
- **Figures, manuscript revisions, and versions:** [Research and figure history](development.md).

The release does not yet include per-run logs, trained weights, all historical ablation configurations, or every plotting script. Recorded comparisons do not substitute for unperformed sub-operation ablations, multi-seed replication, official UAVDT testing, or deployment evaluation. Future additions should associate each new claim with configurations, a code commit, checkpoint selection, and evaluation settings.
