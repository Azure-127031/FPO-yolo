# FPO-YOLO

### 面向特征传递的无人机影像小目标检测器

[English](README.md) | [简体中文](README.zh-CN.md) · [快速开始](#快速开始) · [实验结果](#实验结果) · [代码改进](docs/fpo/architecture.zh-CN.md) · [实验历程](docs/fpo/experiments.zh-CN.md) · [研究与绘图](docs/fpo/development.zh-CN.md)

FPO-YOLO 基于 YOLO11，采用步长为 4、8、16 的 **P2–P4 预测层**，沿同一高分辨率检测路径协调局部特征调制、自适应下采样、跨层上下文聚合与逐位置尺度选择。

在 **VisDrone2019 验证集**上，FPO-YOLO-s 取得 **48.1% mAP50 / 29.2% mAP50–95**，较 YOLO11s 提高 **9.3 / 6.1 个百分点**，参数量减少 **56.2%**。使用验证集选择的同一权重，在 **test-dev** 上取得 **38.9% / 22.7%**。参数量下降伴随估算运算量增加，运行速度还取决于数据集和评估设置。

<p align="center">
  <img src="docs/fpo/assets/architecture.png" alt="FPO-YOLO 整体结构、P2–P4 检测路径及 BCSG 容器" width="100%">
</p>

**结构总览。**BCSG 与 AAD 处理局部特征及尺度转换；CIHF 为高分辨率颈部补充上下文；PASF 将三个颈部输出分别融合至 P2、P3、P4 预测层。[逐层查看代码连接 →](docs/fpo/architecture.zh-CN.md)

## 阅读入口

| 你想了解什么 | 从这里开始 |
|---|---|
| 方法是什么、最终效果怎样 | 本页的[方法概览](#方法概览)与[实验结果](#实验结果) |
| 相对 YOLO11 改了哪些代码、模块如何连接 | [代码改进方案](docs/fpo/architecture.zh-CN.md)：YAML 层级、四个模块的实际计算和模型集成 |
| 每组实验为什么做、结果支持什么选择 | [实验历程](docs/fpo/experiments.zh-CN.md)：E1–E8 对照、训练与续训、固定测试及时间节点 |
| 如何准备环境、训练和评估 | [快速开始](#快速开始) → [完整复现指南](docs/fpo/reproduction.zh-CN.md) |
| 从早期图稿到论文与第二版开源经历了什么 | [研究与绘图历程](docs/fpo/development.zh-CN.md)：时间线、绘图工具、图件修订与前后对照 |

## 更新记录

- **2026-10-11：**同步当前论文的验证/test-dev 结果及完整模型引用，补充 AP75、COCO 尺寸 AP、更新图件及[机器可读结果快照](docs/fpo/data/reported-results.json)。
- **2026-10-10：**精修中英文阅读入口与使用示例，补充同尺度 test-dev 对照和软件引用，保留完整代码、实验与图件演进记录。
- **2026-10-09：**完善第二版中英文文档，加入 UAVDT、AI-TOD-v2 结果及各自评估设置；补全代码改进说明、实验历程、早期时间节点和图件制作过程。
- **2026-09-29：**发布基于 Ultralytics 8.3.237 的 FPO-YOLO 实现。
- **2026-09-08：**建立仓库。

## 方法概览

小目标可用的像素有限。引入浅层预测层能够保留更多空间信息，但这些特征仍需要语义上下文，并在预测时选择合适的尺度。四个组件分别处理这条路径中的不同运算环节：

| 组件 | 在检测路径中的作用 | 实现 |
|---|---|---|
| **BCSG：通道—空间门控瓶颈** | 以级联残差空间门控和通道门控调制局部特征；响应核组由 Scharr 核及均值核初始化，并参与后续学习。 | [fpo_bcsg.py](ultralytics/nn/modules/fpo_bcsg.py) |
| **AAD：自适应聚合下采样** | 在降低分辨率前聚合门控可变形响应，并保留并行常规下采样分支。 | [fpo_aad.py](ultralytics/nn/modules/fpo_aad.py) |
| **CIHF：跨层交互与高分辨率融合** | 通过条件交互更新中高层上下文，再将其对齐、聚合至浅层残差表征。 | [fpo_cihf.py](ultralytics/nn/modules/fpo_cihf.py) |
| **PASF：位置自适应尺度融合** | 对齐颈部输出，在各预测位置分别选择不同尺度的贡献。 | [fpo_pasf.py](ultralytics/nn/modules/fpo_pasf.py) |

BCSG 改变后续阶段可用的特征响应，AAD 改变分辨率降低时这些响应的采样位置。CIHF 将随图像和通道变化的上下文权重引入浅层残差路径；PASF 随后在各预测位置选择尺度，补充 CIHF 空间共享权重无法分别作出的局部选择。上述计算依次连接特征形成、尺度转换、上下文聚合与预测。

**组合证据。**以 CIHF 为共同参考，AAD、BCSG、PASF 单独加入时，验证集 mAP50–95 分别提高 0.4、1.0、0.7 个百分点；完整组合提高 2.6 个百分点，超过三项单独增益之和 2.1。移除任一组件，完整模型下降 0.6–0.9 个百分点，支持所评估配置内的互补性。[逐项查看代码改进 →](docs/fpo/architecture.zh-CN.md) · [组件对照与移除结果 →](docs/fpo/experiments.zh-CN.md#e2-组件组合与移除) · [方法承接关系 →](docs/fpo/development.zh-CN.md#方法承接与改造)

## 实验结果

以下精度单位均为百分比，保留一位小数；提升幅度按舍入前结果计算。mAP50–95 在 IoU 0.50:0.05:0.95 上求平均。**验证集和测试集结果分别列出。**

### VisDrone2019-DET

| 模型 | Val mAP50 | Val mAP50–95 | Test-dev mAP50 | Test-dev mAP50–95 | 参数量（M） | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 32.3 | 18.7 | 26.0 | 14.6 | 2.58 | 6.3 |
| FPO-YOLO-n | 41.2 | 24.6 | 33.0 | 19.1 | 1.34 | 13.5 |
| YOLO11s | 38.8 | 23.2 | 31.7 | 18.3 | 9.43 | 21.6 |
| FPO-YOLO-s | 48.1 | 29.2 | 38.9 | 22.7 | 4.13 | 36.5 |
| FPO-YOLO-m | 51.5 | 31.8 | 41.1 | 24.3 | 14.84 | 121.0 |
| FPO-YOLO-l | 52.2 | 32.4 | 41.4 | 24.5 | 16.50 | 138.0 |

保留官方 train/val/test-dev 划分，test-dev 使用验证集选择的权重。GFLOPs 为 640 像素输入下的分析器估算值，不能直接等同于硬件运行时间。配置另支持 `x` 尺度，本页不汇报未提供的 `x` 实验结果。

<details>
<summary><strong>test-dev 上与同尺度 YOLO11 的对照</strong></summary>

| 模型 | Test-dev mAP50 | Test-dev mAP50–95 | 参数量（M） | GFLOPs |
|---|---:|---:|---:|---:|
| YOLO11n | 26.0 | 14.6 | 2.58 | 6.3 |
| FPO-YOLO-n | **33.0** | **19.1** | **1.34** | 13.5 |
| YOLO11s | 31.7 | 18.3 | 9.43 | 21.6 |
| FPO-YOLO-s | **38.9** | **22.7** | **4.13** | 36.5 |
| YOLO11m | 35.3 | 20.8 | 20.04 | 67.7 |
| FPO-YOLO-m | **41.1** | **24.3** | **14.84** | 121.0 |
| YOLO11l | 36.4 | 21.7 | 25.32 | 87.3 |
| FPO-YOLO-l | **41.4** | **24.5** | **16.50** | 138.0 |

粗体表示同尺度内更高精度或更少参数。精度收益伴随更高估计运算量；所有检查点均在 test-dev 评估前由验证集选择。

</details>

### UAVDT：按完整序列划分的验证集

| 模型 | Val mAP50 | Val mAP50–95 |
|---|---:|---:|
| YOLO11n | 37.0 | 18.7 |
| FPO-YOLO-n | 44.1 | 23.4 |
| YOLO11s | 44.4 | 24.9 |
| FPO-YOLO-s | 52.0 | 28.3 |

自定义划分由 **24 个训练序列 / 6 个验证序列**组成，共 **20,533 / 3,610 张图像**。将官方训练序列名称排序，以随机种子 42 选取完整验证序列；官方测试部分不参与该划分。本表汇报的是**验证集结果**。平均精度提升仍伴随 car、truck 的 AP 指标取舍，以及该数据集上的实测延迟增加。

### AI-TOD-v2：官方测试集

| 模型 | Test mAP50 | Test mAP50–95 |
|---|---:|---:|
| YOLO11n | 34.4 | 14.8 |
| FPO-YOLO-n | 43.1 | 19.5 |
| YOLO11s | 45.5 | 19.9 |
| FPO-YOLO-s | 54.4 | 25.0 |

保留公开的 **11,214 / 2,804 / 14,018** train/val/test 划分。仅以 `train` 训练，使用 `val` 选择权重，以 FP32、`max_det=300` 在 `test` 上评估。总体精度与 AP75 采用 Ultralytics；COCO 尺寸 AP 在下文单列，同样使用 300 个检测结果上限。UAVDT 和 AI-TOD-v2 均**分别训练模型**，不表示 VisDrone 权重的零样本跨域迁移。[各数据集复现设置 →](docs/fpo/reproduction.zh-CN.md#数据集与划分)

### AP75 与目标尺寸结果

严格 IoU 指标采用同一组选定检查点。完整精度值、逐类别 AP、权重哈希及评估设置见[结果 JSON](docs/fpo/data/reported-results.json)。

| 模型 | VisDrone 验证 mAP75 | UAVDT 验证 mAP75 | AI-TOD-v2 测试 mAP75 |
|---|---:|---:|---:|
| YOLO11n | 18.6 | 14.7 | 10.3 |
| FPO-YOLO-n | 25.1 | 21.4 | 14.5 |
| YOLO11s | 23.6 | 24.2 | 14.3 |
| FPO-YOLO-s | 30.2 | 25.7 | 19.2 |

<details>
<summary><strong>COCO 尺寸分组 AP</strong></summary>

**VisDrone2019 validation**

| 模型 | AP_small | AP_medium | AP_large |
|---|---:|---:|---:|
| YOLO11n | 8.7 | 26.9 | 35.2 |
| FPO-YOLO-n | 14.9 | 33.3 | 43.6 |
| YOLO11s | 12.4 | 32.9 | 42.4 |
| FPO-YOLO-s | 18.8 | 38.7 | 44.4 |

**UAVDT validation**

| 模型 | AP_small | AP_medium | AP_large |
|---|---:|---:|---:|
| YOLO11n | 10.5 | 26.1 | 38.0 |
| FPO-YOLO-n | 12.5 | 28.0 | 27.9 |
| YOLO11s | 10.1 | 25.2 | 37.9 |
| FPO-YOLO-s | 12.7 | 39.7 | 35.6 |

**AI-TOD-v2 test**

| 模型 | AP_verytiny | AP_tiny | AP_small | AP_medium |
|---|---:|---:|---:|---:|
| YOLO11n | 3.5 | 12.7 | 20.9 | 28.2 |
| FPO-YOLO-n | 7.1 | 18.0 | 25.2 | 31.6 |
| YOLO11s | 4.7 | 17.1 | 26.8 | 37.0 |
| FPO-YOLO-s | 10.3 | 23.2 | 31.7 | 38.8 |

尺寸 AP 使用原图面积、COCOeval 和 maxDets=300。VisDrone/UAVDT 分界为32²/96²，AI-TOD-v2 为8²/16²/32²；这些数值与 Ultralytics AP 分别报告。

</details>

### 受控设计对照

[E1–E8 实验索引](docs/fpo/experiments.zh-CN.md)将各项设计选择对应到具体对照：预测层与 CIHF、组件组合与移除、采样变体、BCSG 响应组织、融合检测头、回归损失、模型尺度，以及分别训练的补充数据集。该索引区分整体结构收益与组件增益，并汇报相应资源开销。

## 定性比较

<p align="center">
  <img src="docs/fpo/assets/qualitative.png" alt="四个 VisDrone 场景中 YOLO11s 与 FPO-YOLO-s 的检测及检测条件 Grad-CAM 对照" width="100%">
</p>

四行图像分别比较车辆区域中的检测结果与检测条件 Grad-CAM。每个区域内，两模型的 CAM 使用同一组真实车辆作为条件。这些图用于展示预测差异，不能代替组件作用的定量验证。图像来自 VisDrone，检测输出和可视化由本项目生成。[可视化制作方法 →](docs/fpo/development.zh-CN.md#检测与响应可视化)

## 快速开始

### 1. 安装本仓库与编译算子

参考环境为 **Python 3.10、PyTorch 2.4.1、CUDA 11.8、Ultralytics 8.3.237**。先安装适合本机的 PyTorch/CUDA 组合，再在独立环境中执行：

```bash
git clone https://github.com/Azure-127031/FPO-yolo.git FPO-YOLO
cd FPO-YOLO
python -m pip install -e .
python -m pip install -r requirements-fpo.txt
```

AAD 依赖**包含编译算子的 MMCV**，需要 `mmcv.ops.ModulatedDeformConv2d`，`mmcv-lite` 无法替代。MMCV 应与 PyTorch、CUDA 匹配；没有对应 wheel 时，按 [MMCV 官方安装说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)构建。检查实际导入的是当前仓库：

```bash
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
python -c "from mmcv.ops import ModulatedDeformConv2d; print('MMCV ops import OK')"
python scripts/check_model.py --scale s
```

这些检查分别确认 Python 加载本仓库、编译算子可以导入，以及所选模型结构能够构建。

### 2. 准备 YOLO 格式数据

修改 [configs/visdrone.yaml](configs/visdrone.yaml) 的 `path`，指向数据根目录：

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

每张图像对应同名标签文件，每行格式为 `class_id x_center y_center width height`，坐标归一化至 `[0, 1]`。类别顺序须与 YAML 一致。请从数据集作者处获取数据；当前版本不分发完整数据集或转换工具。[划分及标注要求 →](docs/fpo/reproduction.zh-CN.md#数据集与划分)

### 3. 从随机初始化训练

```bash
python scripts/train.py --scale s --data configs/visdrone.yaml --batch 12 --device 0
```

公开训练入口从 YAML 初始化，使用 `imgsz=640`、最多 200 轮、SGD、patience100、AMP、`close_mosaic=10` 及 seed0。已报告 VisDrone FPO-n/s 使用 `--batch 12`，UAVDT 使用 `--batch 16`，上例对应 VisDrone。脚本仍默认16，并支持显式覆盖批大小。此入口对应 VisDrone/UAVDT 设置；**AI-TOD-v2 应使用单独的设置**。

### 4. 评估验证集选择的权重

```bash
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split val
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split test --save-json
```

该 YAML 中 `--split test` 对应 `images/test-dev`。若 Ultralytics 为运行目录添加后缀，应使用实际路径。测试前固定权重选择，不用 `last.pt` 替换 `best.pt`。

`scripts/val.py` 使用 FP16 和 batch 16；AI-TOD-v2 的 FP32/batch-15 设置见[对应 API 示例](docs/fpo/reproduction.zh-CN.md#ai-tod-v2-训练与测试)。

### 5. 使用训练权重进行预测

```python
from ultralytics import YOLO

model = YOLO("runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt")
model.predict(source="path/to/image.jpg", imgsz=640, conf=0.25, device=0, save=True)
```

将路径替换为自己的权重和图像。此处 `conf=0.25` 用于过滤展示预测；定量评估采用[复现指南](docs/fpo/reproduction.zh-CN.md)中的设置，包括 0.001 置信度过滤阈值。当前版本不提供预训练权重。

## 研究与图件演进

从早期模块草图到第二版开源，项目经历了**分模块绘图、整体路径整理、代码与实验核对、补充数据集评估和论文图件精修**。研究过程和图的制作过程同步推进：一张结构图既要解释数据流，也要与实际实现对应。

<details>
<summary><strong>研究时间线：从早期图稿到第二版开源</strong></summary>

| 时间 | 主要进展 |
|---|---|
| 2026 年 3—7 月¹ | 保存 YOLO11 基线、BCSG、AAD、SBA/ASFF 等模块图，迭代 PowerPoint 源稿，积累 VisDrone 数据与场景图。 |
| 2026 年 8 月¹ | 整理 FPO 总体结构、基线和模块细图的 PPTX/PDF 版本。 |
| 2026 年 9 月 3—16 日 | 汇集初稿、代码和日志；连续核对方法、消融、类别结果及图表，补充 UAVDT 章节。9 月 8 日建立仓库。 |
| 2026 年 9 月 18—29 日 | 整理英文双栏稿与实验协议，将 BCSG 三面板裁剪对齐；9 月 29 日首次发布实现。 |
| 2026 年 10 月 1—6 日 | 梳理补充实验协议，完成 AI-TOD-v2 数据审计、四组训练及固定测试集评估。 |
| 2026 年 10 月 7—8 日 | 纳入补充结果、回应批注；核对结构图语义，保留原有形状和走线，逐模块换色并整理实验图。 |
| 2026 年 10 月 9 日 | 发布中英文第二版文档，补充论文开源入口，串联更早的研究与绘图记录。 |

¹ 早期月份依据保留图稿的文件时间，表示图稿版本，不推断实验完成日。完整时间线逐项说明日期来源。

</details>

图件制作包括 **PowerPoint → PDF** 的结构图导出、**LaTeX/TikZ** 的 BCSG 三面板装配、**Matplotlib** 的定量图，以及实际预测与检测条件 Grad-CAM 的面板整理。详细页面保留配色前后对照，并解释每次调整解决了什么问题。

**[阅读完整研究与绘图历程 →](docs/fpo/development.zh-CN.md)** · [直接查看绘图方法](docs/fpo/development.zh-CN.md#图件如何形成) · [Read in English](docs/fpo/development.md)

## 仓库结构

| 路径 | 内容 |
|---|---|
| [ultralytics/cfg/models/11/yolo11-fpo.yaml](ultralytics/cfg/models/11/yolo11-fpo.yaml) | 共用结构及 n/s/m/l/x 尺度定义 |
| [ultralytics/nn/modules](ultralytics/nn/modules) | FPO 模块与继承的 Ultralytics 组件 |
| [scripts](scripts) | 结构检查、训练和评估入口 |
| [configs](configs) | VisDrone、UAVDT 数据模板 |
| [docs/fpo](docs/fpo) | 代码改进、实验历程、复现指南、研究与绘图记录及图件资源 |

## 发布内容与适用范围

当前版本包含实现代码、结构与数据配置、运行入口和文档，**不包含训练权重及完整原始实验日志**。展示图件是已有研究图的导出资源，不是可由权重重新生成全部可视化的完整工具包。

自定义可变形算子需要兼容的编译环境。本版本不据此声称已验证 ONNX/TensorRT 导出、部署速度或任意依赖版本组合。FPO-YOLO 减少了参数量，但不代表各尺度 GFLOPs 均下降，或每个类别指标均提升。

## 致谢、引用与许可

代码基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 和 [MMCV](https://github.com/open-mmlab/mmcv)。AAD 使用修改后的 DCNv2 采样，CIHF 承接 SBA 式交互，PASF 采用 ASFF 式空间尺度加权与坐标注意力式方向细化。[演进记录](docs/fpo/development.zh-CN.md#方法承接与改造)区分这些来源与本项目的具体修改。

对应论文题为 **FPO-YOLO: A Feature Propagation-Oriented Detector for Small Objects in UAV Imagery**。当前发布版本未提供正式发表元数据；正式论文引用信息公布前，可使用以下软件引用并记录实验使用的 commit。该条目引用代码仓库，不代表已发表论文。

```bibtex
@misc{fpoyolo2026code,
  title = {FPO-YOLO: Open-source implementation},
  year = {2026},
  howpublished = {\url{https://github.com/Azure-127031/FPO-yolo}}
}
```

仓库遵循 [AGPL-3.0 许可](LICENSE)，请保留上游声明。提交问题时请附 commit、环境版本、数据 YAML、运行命令及完整报错，不附私人数据或凭据。
