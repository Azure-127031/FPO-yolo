# 复现指南

[项目首页](../../README.zh-CN.md) · [English](reproduction.md) · [代码改进](architecture.zh-CN.md) · [实验历程](experiments.zh-CN.md) · [研究与图件演进](development.zh-CN.md)

公开实现基于 Ultralytics 8.3.237。本指南区分运行入口的默认设置与结果对应的数据集设置。数据需自行准备；当前版本不包含下载好的数据集、标注转换工具或训练权重。

## 环境与安装

在独立 Python 环境中安装兼容的 PyTorch/CUDA。参考组合为 Python 3.10、PyTorch 2.4.1、CUDA 11.8；VisDrone/UAVDT 延迟实验使用 NVIDIA L20。

```bash
git clone https://github.com/Azure-127031/FPO-yolo.git FPO-YOLO
cd FPO-YOLO
python -m pip install -e .
python -m pip install -r requirements-fpo.txt
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
python -c "from mmcv.ops import ModulatedDeformConv2d; print('MMCV ops import OK')"
python scripts/check_model.py --scale s
```

`ultralytics.__file__` 应指向本仓库。AAD 在加载时导入 MMCV 的编译算子；`mmcv-lite` 或与当前 PyTorch/CUDA 不匹配的 wheel 无法提供可用运行环境。具体构建见 [MMCV 官方安装说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。

依赖文件要求 `mmcv>=2.0.0,<2.3.0`，对应二进制包是否可用取决于平台和软件组合。

## 数据集与划分

| 数据集 | 训练 / 验证 / 最终评估 | 类别数 | 关键区别 |
|---|---|---:|---|
| VisDrone2019-DET | 官方 train / val / test-dev | 10 | 先以 val 选择权重，再以固定权重评估 test-dev。 |
| UAVDT | 官方训练部分中的 24 / 6 个完整序列；20,533 / 3,610 张图像 | 3 | 结果来自自定义验证集，不是官方测试集。 |
| AI-TOD-v2 | 公开 train / val / test；11,214 / 2,804 / 14,018 张图像 | 8 | 只使用 train 训练，不合并 train 与 val。 |

各数据集分别训练模型，考察重新训练后的结构适用性，不属于零样本跨域迁移。

### YOLO 标签

图像与标签文件同名，每行标签为：

```text
class_id x_center y_center width height
```

类别从 0 编号，坐标按图像宽高归一化。标签映射须与 YAML 类别顺序一致。检查目标框位于图像内且面积大于零，保留合法无目标图像，并遵循数据集对排除/忽略标注的处理。VisDrone 的忽略区域及非目标类别不能映射为十个目标类别。

修改 [configs/visdrone.yaml](../../configs/visdrone.yaml)、[configs/uavdt.yaml](../../configs/uavdt.yaml) 的 `path` 后使用。二者 `test` 对应不同子集：VisDrone 指向 test-dev，UAVDT 模板预留独立官方测试目录。存在 test 路径不等于已经汇报测试结果。

UAVDT 先对官方训练序列名称排序，以 seed 42 无放回选取 6 个完整验证序列，其余 24 个训练；转换时保留序列归属，官方 20 个测试序列独立存放。只核对总图像数不足以复现划分，应保存实际序列名单及数据清单。

AI-TOD-v2 使用 **v2 标注**制作 YAML，保留公开子集归属和类别映射。转换时将框裁剪至图像范围，去除零面积框，保留无目标图像，不能替换为原始 AI-TOD 标注。当前版本无自动 AI-TOD 转换器及模板，应先准备、审计数据 YAML，再使用下方示例。

## 训练与评估设置

| 设置 | VisDrone / UAVDT | AI-TOD-v2 |
|---|---|---|
| 初始化 | 模型 YAML / 随机初始化 | 模型 YAML / 随机初始化 |
| 输入尺寸 | 640×640 | 640×640 |
| 训练 batch | VisDrone YOLO11n/s：16，FPO-n/s：12；UAVDT：16 | 15 |
| 最大轮数 | 200 | 200 |
| 早停 patience | 100 | 50 |
| 优化器 | SGD | `auto` |
| SGD 参数 | lr0=0.01、lrf=0.01、momentum=0.937、weight_decay=0.0005 | 由公开训练器选择 |
| Warm-up | 3 轮 | 训练器默认 |
| 学习率变化 | 线性，`cos_lr=False` | 训练器默认 |
| 训练 AMP | 开启 | 开启 |
| Mosaic 关闭 | 最后 10 轮 | 最后 10 轮 |
| Seed | 已核验主要 n/s 实验使用 0 | 汇报实验使用 0 |
| 评估精度 | FP16 | FP32 |
| 评估 batch | 16 | 15 |
| 置信度过滤 | 检测评估器默认 0.001 | 显式设为 0.001 |
| NMS IoU / 最大检测数 | 0.7 / 300 | 0.7 / 300 |
| 测试时增强 | 关闭 | 关闭 |

已核验主要 n/s 实验使用 seed0。`scripts/train.py` 支持 `--batch`，VisDrone FPO-n/s 对应12，UAVDT 对应16；不将这些设置追溯至未记录的其他历史实验。结构对照保留 CIoU 回归目标，FPO-YOLO 沿用分类与分布焦点损失训练流程，不因这些模块另行引入自定义训练损失。

### VisDrone 与 UAVDT 运行入口

```bash
python scripts/train.py --scale s --data configs/visdrone.yaml --batch 12 --device 0 --project runs/fpo_yolo --name fpo_yolo_s_e200
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split val
python scripts/val.py --weights runs/fpo_yolo/fpo_yolo_s_e200/weights/best.pt --data configs/visdrone.yaml --device 0 --split test --save-json
```

UAVDT 训练使用 `--data configs/uavdt.yaml --batch 16`，表中结果使用 `--split val`。重复运行可能获得带后缀目录，必须使用本次实际权重路径。

训练入口支持 n/s/m/l/x。已汇报结果覆盖 VisDrone 的 n/s/m/l，以及 UAVDT、AI-TOD-v2 的 n/s；支持配置不代表其他尺度已有实测结果。

### AI-TOD-v2 训练与测试

**先制作并审计 AI-TOD-v2 YAML**，再在安装好的仓库环境中运行以下 Python 示例。`scale` 可为 `n` 或 `s`；对应基线使用 `YOLO(f"yolo11{scale}.yaml")`，其他训练/评估参数保持一致。

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

# 使用训练器实际目录，包括可能自动添加的后缀。
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

当前仓库的检测 fitness 使用验证集 mAP50–95，定义见 `ultralytics/utils/metrics.py`，据此选择 `best.pt`。测试集不能参与权重或超参数选择。四组已汇报 AI-TOD-v2 实验均完成 200 轮。

本页 AI-TOD-v2 结果采用共同 Ultralytics 评估器，不能与 trainval 训练、每图最多 1,500 检测或独立 tiny 尺寸 COCO 评估的结果直接等同。AP75 和单独计算的 COCO 尺寸 AP 已加入[结果快照](data/reported-results.json)。尺寸分析采用 maxDets=300，与官方1,500检测结果协议分别说明。

## 对齐公开结构

- **BCSG：**`C3k2_BCSG` 控制瓶颈 shortcut，不切换 C3k 分支；可学习核组按连续输出通道分段，不能解释为按通道交错的微分核组。
- **AAD：**实际调用 `ModulatedDeformConv2d`（DCNv2），历史类名 `MDCv3` 不表示算子版本。
- **CIHF：**YAML 中对应 `ScalSeq_SBA`，即公开的 CIHF 实现。
- **PASF：**使用 `Detect_PASF_Final`，不以历史 `Detect_ASFF1`/ASFF 检测头或其他结构的权重替换。

比较模型时不能悄然替换编译算子或检测头。训练时类别数来自数据 YAML，因此不同数据集的参数量可能略有差别。

## 保存可复现记录

保存代码 commit、环境版本、数据/序列清单及哈希、实际 `args.yaml`、`results.csv`、训练日志、验证集选择的权重和评估输出。基线与本文模型分别存储，在各数据集内使用一致设置。

运行时间比较须记录 GPU、精度、batch、输入尺寸及预处理/推理/后处理时间。分析器对自定义算子和普通卷积的计数覆盖可能不同，GFLOPs 与实测延迟是两个量；不能将单次计时写成普遍部署速度结论。

## 常见问题

| 现象 | 优先检查 |
|---|---|
| `No module named mmcv._ext` 或缺少可变形算子 | 安装与 PyTorch/CUDA 匹配的含编译算子 MMCV，必要时清理同一环境中冲突的 `mmcv-lite`。 |
| 解析器不认识 FPO 模块 | 检查 `ultralytics.__file__`，可能被另一安装版本遮蔽。 |
| 权重形状/类别不匹配 | 核对尺度、类别顺序、最终 PASF 结构是否与权重一致。 |
| 测试结果不同 | 核对标注版本、子集成员、验证集选择的权重、数值精度、conf、NMS、最大检测数及评估器。 |
| 新运行目录带后缀 | 使用训练器实际 `save_dir`，避免评估旧运行权重。 |

本指南记录发布版本及汇报设置，不在缺少原始清单、权重时声称全部历史实验可精确重建。
