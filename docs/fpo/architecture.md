# Code design and implementation guide

[Project home](../../README.md) · [简体中文](architecture.zh-CN.md) · [Experiment history](experiments.md) · [Research and figures](development.md) · [Reproduction](reproduction.md)

This page explains the **implemented architectural changes**: their reference points, code locations, integration, and supporting experiments. It follows the released source and retained manuscript records. The order below follows network functions, not an inferred chronology of training runs.

## Design route and verification questions

YOLO11s supplies the detection framework. FPO-YOLO changes the finest prediction stride from 8 to 4, then addresses local representation, downsampling, context, and scale selection along that high-resolution path.

| Stage | Question | Implemented change | Evidence |
|---|---|---|---|
| Prediction levels | How can a small object retain a finer spatial grid? | P3/P4/P5 → P2/P3/P4 | [E1 resolution and fusion](experiments.md#e1-resolution-and-fusion) |
| Local representation | How should spatial and channel responses be modulated? | BCSG group convolution and cascaded residual gates | [E2 combinations](experiments.md#e2-component-combinations-and-removal), [E4 gate organization](experiments.md#e4-bcsg-response-organization) |
| Scale transitions | How can sampling adapt to the input? | AAD regular branch plus gated deformable aggregation | [E3 sampling variants](experiments.md#e3-aad-sampling-variants) |
| Cross-level context | How can shallow features receive deeper context? | CIHF conditional interaction followed by aggregation at P2 | [E1 interaction comparison](experiments.md#e1-resolution-and-fusion) |
| Prediction fusion | How can different locations select different scales? | PASF alignment and spatial scale weighting at each output level | [E5 head replacements](experiments.md#e5-head-replacements) |
| Regression objective | Are architecture changes distinct from loss replacement? | Retain CIoU, classification loss, and DFL | [E6 regression losses](experiments.md#e6-regression-losses) |

## How the configuration connects

The entry point is [`yolo11-fpo.yaml`](../../ultralytics/cfg/models/11/yolo11-fpo.yaml). Indices below are zero-based YAML layer indices. Channels follow the n/s/m/l/x scaling rules; the data YAML supplies the actual class count.

| Layers | Output or operation | Role |
|---|---|---|
| 0 | 3×3 stride-2 Conv | Stem at P1 resolution |
| 1–2, 3–4, 5–8 | AAD and C3k2_BCSG; SPPF/C2PSA at the end | B2, B3, B4; deepest backbone level P4 |
| 9 | `ScalSeq_SBA([B2, B3, B4])` | CIHF high-resolution aggregation |
| 10–12 | Concatenate B2, project with 1×1 Conv, apply C3k2_BCSG | N2 |
| 13–16 | AAD, concatenate B3, project and locally refine | N3 |
| 17–20 | AAD, concatenate B4, project and locally refine | N4 |
| 21 | `Detect_PASF_Final([N2, N3, N4])` | P2/P3/P4 predictions |

For scale s at 640×640, B2/B3/B4 have shapes 128×160×160, 256×80×80, and 256×40×40; N2/N3/N4 are 64×160×160, 128×80×80, and 256×40×40, excluding batch. See the [diagram and code checks](development.md#figure-development).

## BCSG from multiplicative gating to residual modulation

The version notes in [`fpo_bcsg.py`](../../ultralytics/nn/modules/fpo_bcsg.py) identify three changes from BCSG-v1 to BCSG-v2. This module version is separate from the second edition of the repository documentation.

| Earlier operation | Current implementation | Design intent |
|---|---|---|
| 3×3 depthwise convolution | Group convolution with a target group width of about 32 channels and a divisible group count | Retain local interactions within channel groups |
| Direct multiplication by a spatial gate | `F_sp = F + sigmoid(alpha_sp) * F * g_sp` | Modulate an increment while retaining the base feature |
| Both gates estimated from the original F | Estimate channel gates from F_sp, then add another residual increment | Let channel selection use the spatially modulated representation |

Both raw increment parameters start at −1.5, giving about 0.182 after sigmoid. The spatial response bank starts from Scharr-X, Scharr-Y, and mean kernels but remains learnable. Its outputs are split into contiguous ranges; the documentation therefore describes contiguous partitions rather than an independently paired fixed derivative bank for every input channel.

`C3k2_BCSG` splits features, collects serial bottleneck outputs, concatenates them, and applies pointwise fusion. Its Boolean argument controls **internal bottleneck shortcuts**; False layers still use BCSG. The response-organization comparison supports the current partitioned variant, but does not separately measure each of these three v1→v2 changes.

## AAD aggregates deformable responses before downsampling

In [`fpo_aad.py`](../../ultralytics/nn/modules/fpo_aad.py), the two `AAD_v2` branches each produce half the target channels. A regular branch uses stride-2 convolution; the dynamic branch performs modulated deformable convolution at the original resolution, then average pooling. A learned sigmoid scalar weights the two branches before concatenation and 1×1 fusion.

Despite its historical name, `MDCv3` calls **MMCV `ModulatedDeformConv2d`, the DCNv2 operator**. It separately predicts two offset components and a modulation mask, with offsets constrained as follows:

```text
offset = tanh(predicted_offset) * abs(max_offset) * gate
gate shape = B × 1 × 1 × 1
```

The amplitude gate is one shared scalar per sample, not an independent map at every pixel. `max_offset` is learnable. The defined `res_scale` does not participate in the current forward return path, so it is not described as an active residual branch. Comments claiming parameter halving or improved stability are not treated as independently verified results; [E3](experiments.md#e3-aad-sampling-variants) evaluates the complete replacement.

## CIHF interacts before high-resolution aggregation

[`fpo_cihf.py`](../../ultralytics/nn/modules/fpo_cihf.py) separates two stages:

1. `SBA_Precise` projects B3/B4 into a shared width, generates conditional transforms and mixing parameters from a deep descriptor, exchanges information in both directions, and returns a refined middle-level feature.
2. `ScalSeq_SBA` projects B2, the refined middle feature, and the original B4 onto P2 resolution. `HyperScaleDynamicFusionAllScale` uses global descriptors of all three aligned features to produce scale/channel weights, then adds the aggregation to the projected B2 path.

The code records a change from a single gate-derived descriptor to an all-scale descriptor in stage two. Current scale weights have shape B×C×3×1×1; the other modulation has shape B×C×1×1. Both are spatially shared, regardless of the variable name `spatial_gate`. Scale weights use sigmoid, not a scale-normalized Softmax.

The current configuration uses 128 channels for interaction and aggregation; the parser does not apply the model width multiplier again to `ScalSeq_SBA`'s output width. Available comparisons quantify aggregation without interaction versus full CIHF. They do not supply an isolated gain for changing the descriptor source.

## PASF from spatial scale weighting to an isolated final head

[`fpo_pasf.py`](../../ultralytics/nn/modules/fpo_pasf.py) retains ASFF-style alignment and spatial scale selection while isolating the paper-final implementation as `Detect_PASF_Final` to avoid resolving a model name to a historical head.

| Step | Current code |
|---|---|
| Alignment | `PureAlign`: 1×1 projection followed by average pooling, nearest-neighbor upsampling, or unchanged resolution |
| Weight generation | Sum the three aligned inputs; map through 16 hidden channels, three output channels, and BatchNorm |
| Normalization | Divide logits by the absolute learned temperature and apply Softmax across three scales; weights vary spatially and are shared over channels |
| Initialization | Zero-initialize the three-logit convolution so initial scale weights are equal |
| Refinement | Convolution, directional attention, and projection; combine with the shortcut using a fixed coefficient of 0.5 |
| Prediction | Fuse independently for P2/P3/P4, then apply classification/regression branches and 16-bin DFL decoding |

`beta=0.5` is a buffer, not a learned parameter. The temperature uses `abs()` without an additional positive lower clamp in the current implementation. These are source-code observations, not changes made in this documentation update. Head-replacement results support PASF as a whole, not separate attribution to initialization, temperature, or directional attention.

## From module files to a runnable model

Copying four modules alone does not complete integration:

| File | Responsibility |
|---|---|
| [`modules/__init__.py`](../../ultralytics/nn/modules/__init__.py) | Export FPO module entry points |
| [`nn/tasks.py`](../../ultralytics/nn/tasks.py) | Parse AAD/BCSG channels and repeats; pass multi-input channel lists to CIHF/PASF |
| [`yolo11-fpo.yaml`](../../ultralytics/cfg/models/11/yolo11-fpo.yaml) | Specify modules, connections, shortcuts, and scales |
| [`check_model.py`](../../scripts/check_model.py) | Construct a selected scale and inspect the structure |
| [`train.py`](../../scripts/train.py), [`val.py`](../../scripts/val.py) | Training and evaluation entry points; see the reproduction guide for dataset-specific settings |

A useful reading order is YAML → module forward → parser arguments → experiment comparison. Historical ASFM/SBA, Detect/ASFF, and MDCv3 names require their original configuration context. Reproduce the released architecture with its current YAML classes rather than substituting checkpoints by similar names.

Continue with [experimental evidence](experiments.md) or [running the repository](reproduction.md).
