# Research and figure development

[Project home](../../README.md) · [简体中文](development.zh-CN.md) · [Reproduction guide](reproduction.md)

FPO-YOLO studies how a high-resolution detection path uses local detail after successive feature transformations. This record explains the design sequence, the evidence used to examine component complementarity, and how the manuscript figures were produced. Dates below refer to repository commits or dated figure/document revisions; they are not inferred training dates.

## From a shallow prediction level to a coordinated path

The starting point is the limited spatial support of aerial small objects. Moving prediction from P3–P5 to P2–P4 retains a finer grid, but does not by itself provide the shallow features with enough context. The structural comparisons distinguish these steps:

| Configuration | VisDrone val mAP50 (%) | mAP50–95 (%) |
|---|---:|---:|
| YOLO11s | 38.8 | 23.2 |
| P2–P4 prediction | 42.8 | 25.9 |
| Shallow aggregation without cross-level interaction | 43.1 | 26.1 |
| Complete CIHF reference | 43.9 | 26.6 |

CIHF establishes the common high-resolution reference. The remaining components address operations that its cross-level aggregation does not replace:

1. **BCSG modifies the local representation.** A learnable response bank drives spatial modulation, followed by channel modulation. It does not change the downsampling grid.
2. **AAD modifies the scale transition.** Deformable aggregation operates before resolution reduction, alongside regular sampling. It addresses a different operation from local residual gating.
3. **PASF modifies prediction-side scale selection.** CIHF's scale/channel aggregation weights are shared across spatial locations; PASF instead calculates location-dependent weights for aligned neck features at each prediction level.

The resulting model connects feature formation, resolution change, context exchange, and prediction-side selection on the same path. The rationale is examined through comparisons rather than inferred from the names of the modules.

## What the component combinations show

The following configurations use the same CIHF reference and the same VisDrone training protocol:

| Enabled components | mAP50 (%) | mAP50–95 (%) |
|---|---:|---:|
| CIHF | 43.9 | 26.6 |
| CIHF + AAD | 44.6 | 27.0 |
| CIHF + BCSG | 45.6 | 27.6 |
| CIHF + PASF | 45.0 | 27.3 |
| CIHF + AAD + BCSG | 46.5 | 28.3 |
| CIHF + BCSG + PASF | 46.6 | 28.6 |
| CIHF + AAD + PASF | 46.4 | 28.3 |
| CIHF + AAD + BCSG + PASF | 47.9 | 29.2 |

Relative to CIHF, the isolated mAP50–95 improvements are 0.4, 1.0, and 0.7 points. Their sum is 2.1 points, whereas the full model gains 2.6 points. Removing AAD, BCSG, or PASF from the full configuration reduces mAP50–95 by 0.6, 0.9, and 0.9 points, respectively. These observations support non-additive, complementary effects **within this tested configuration**. They do not constitute multi-seed significance testing or a full four-factor interaction analysis.

## Design lineage

The code retains established operators where appropriate and modifies how they are used in this high-resolution detector:

| Component | Antecedent | Implemented distinction |
|---|---|---|
| BCSG | Group convolution, Scharr/mean initialization, spatial and channel gating | A learnable response bank drives cascaded residual gates in the bottleneck. Initialization describes the starting kernels, not fixed derivatives throughout training. |
| AAD | Modulated deformable convolution (DCNv2) | Separately predicted offsets, bounded sample-wise offset modulation, deformable aggregation before pooling, and a parallel regular branch with fusion. The historical class name `MDCv3` does **not** identify the DCNv3 operator. |
| CIHF | SBA interaction from DuAT | Conditional transformations of the interacting middle/deep features, followed by aligned scale/channel aggregation onto the shallow residual path. |
| PASF | ASFF and coordinate attention | Scale alignment and spatial Softmax weighting are retained; the released head adds its weight-generation, normalization, temperature, and directional residual-refinement design. |

These are architectural modifications and their coordinated use, rather than claims that all underlying operators were invented here. Refer to the module files linked in the [README](../../README.md#method) for the implementation.

## Documented milestones

| Date | Recorded development |
|---|---|
| 2026-09-08 | Initial repository commit (`366a29d`). |
| 2026-09-29 | Public implementation release (`9016224`); dated BCSG panel assembly aligns the bottleneck, spatial gate, and channel gate at a common visible width. |
| 2026-10-07 | Manuscript revision incorporates the AI-TOD-v2 test protocol and detection-conditioned qualitative comparison. |
| 2026-10-08 | Architecture figure is reconciled with the implemented stride/shortcut semantics; original geometry and connectors are retained, and module-specific colors are revised. The accuracy–complexity comparison is organized by mAP50. |
| 2026-10-09 | English/Chinese manuscripts are synchronized, nearby float placement is revised, and this research/documentation record is prepared. |

## Figure development

### Architecture: revise the original, check against code

The architecture drawing starts from the project's original **PowerPoint diagram exported to PDF**. Revisions preserve the established layout, block shapes, feature-plane geometry, and connector routes. The implementation—not the visual convenience of a schematic—determines the labels and scale changes.

The code checks cover:

- **Resolution transitions.** The initial stem is a 3×3 convolution with stride 2; the subsequent AAD stages reduce resolution by 2. A stride-1 legend must not describe a reduction in spatial dimensions.
- **Feature names and shapes.** Backbone features use B2/B3/B4; neck outputs use N2/N3/N4. For the `s` configuration and a 640×640 input, backbone outputs are 128×160×160, 256×80×80, and 256×40×40; neck outputs are 64×160×160, 128×80×80, and 256×40×40, respectively. Shapes exclude the batch dimension.
- **Container semantics.** In `C3k2_BCSG`, the Boolean argument controls each internal BCSG shortcut. The released container does not switch between C3k=False/True structural branches.
- **Head connections.** The three neck features feed the released `Detect_PASF_Final` head, producing P2/P3/P4 predictions.

The revised palette replaces individual fills and relevant gradient endpoints. Convolution blocks use muted blue, splitting uses teal, BCSG containers use lavender, concatenation uses warm sand, and distinct attention/fusion blocks retain consistent color identities. This is an object-level color revision, not an overlay across the whole page.

![Original palette and revised module colors, side by side](assets/architecture-palette-evolution.png)

*Palette revision of the original diagram. The older panel is retained only to document visual development; use the final architecture below for the released structure.*

[Final overview PDF](assets/architecture.pdf) · [Browser preview](assets/architecture.png)

### BCSG: assemble the computation, not just three named boxes

The BCSG figure combines three cropped source panels: the bottleneck, the spatial gate, and the channel gate. **LaTeX/TikZ** places the exported panels at a shared visible width and aligns labels (a), (b), and (c) with their content. The drawing shows residual modulation, ordered output slicing of the response bank, and local/global channel processing.

![BCSG bottleneck, spatial gate, and channel gate](assets/bcsg.png)

The response-bank illustration follows the code's contiguous output-channel partitions. The diagram should not be read as a separate Scharr-X/Scharr-Y pair independently applied to every channel. [BCSG figure PDF](assets/bcsg.pdf)

### Experimental plots: tie the picture to the result table

Quantitative plots are prepared with **Python/Matplotlib** and exported to PDF for the manuscript. Class-distribution counts are preserved, as is the source bounding-box size-density panel. Layout changes do not generate a replacement distribution.

The accuracy–complexity bubble plot uses **parameters on the horizontal axis, mAP50 on the vertical axis, and GFLOPs as bubble area**. Model numbering and legend order follow the mAP50 ordering. Points retain their reported values; a layout change does not alter an experimental result. Results taken from other papers remain reported comparisons with different possible training conditions.

![Accuracy–complexity comparison using mAP50](assets/accuracy-complexity.png)

[Plot PDF](assets/accuracy-complexity.pdf)

## Detection and response visualization

Detection panels are generated from actual model predictions, using the same region of interest to compare YOLO11s and FPO-YOLO-s. The Grad-CAM target is conditioned on the **same annotated vehicle group within each region**:

1. Select decoded candidate predictions near each target vehicle and use its class score.
2. Backpropagate the sum of the selected scores through all three classification branches.
3. Obtain channel weights by spatially averaging the activation gradients and apply ReLU to the weighted activation sum.
4. Resize the level CAMs, sum them, remove letterbox padding, and map the result back to the source image.
5. Normalize the resulting CAM for display and assemble paired detection/response panels using a consistent layout.

This target construction retains each detector's actual prediction levels, including FPO-YOLO's P2 branch. A display-normalized heatmap is not an absolute response-magnitude comparison across models. The images show selected scenes and should be interpreted together with aggregate and class-wise results.

[Qualitative figure PDF](assets/qualitative.pdf) · [Browser preview](assets/qualitative.png)

## Figure assets and reproducibility

| Asset | Source workflow | Released form |
|---|---|---|
| Architecture | Original PowerPoint → PDF; code/label checks and object-level color revision | PDF + PNG |
| BCSG | Exported panels → tight crop → LaTeX/TikZ assembly | PDF + PNG |
| Accuracy–complexity | Result table → Matplotlib | PDF + PNG |
| Qualitative comparison | Predictions and detection-conditioned Grad-CAM → Matplotlib panel assembly | PDF + PNG |
| Palette evolution | Before/after exports of the original architecture | PNG |

PNG files support GitHub viewing; PDFs retain the manuscript exports. Exported PDFs can contain embedded image panels and are not all pure vector drawings. The repository does not yet include all source slide files, historical visualization scripts, model checkpoints, or dataset images required to recompute these assets.

The development principle is to revise structure where the code demands it, revise appearance at the individual-object level, and keep the underlying evidence unchanged.
