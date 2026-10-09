# 代码改进方案与实现导读

[项目首页](../../README.zh-CN.md) · [English](architecture.md) · [实验历程](experiments.zh-CN.md) · [研究与绘图历程](development.zh-CN.md) · [复现指南](reproduction.zh-CN.md)

本页说明 FPO-YOLO **已经实现的结构改进**：相对什么改变、改动落在哪里、怎样进入完整网络，以及有哪些实验支持。说明依据本仓库源码和现有论文记录；各小节按网络功能组织，不把这一阅读顺序当作历史训练顺序。

## 改进路线与验证问题

YOLO11s 提供基础检测框架。FPO-YOLO 将最细预测步长从 8 调整为 4，再围绕这条高分辨率路径处理局部表征、下采样、上下文和尺度选择。各项改动不是同一个位置上的重复注意力操作。

| 环节 | 要解决的问题 | 已实现方案 | 对应实验 |
|---|---|---|---|
| 预测层级 | 小目标在粗网格上可用的位置有限 | 从 P3/P4/P5 转为 P2/P3/P4 | [E1 高分辨率与融合结构](experiments.zh-CN.md#e1-高分辨率与融合结构) |
| 局部表征 | 局部响应需要空间和通道两方面的调制 | BCSG 分组卷积与级联残差门控 | [E2 组件组合](experiments.zh-CN.md#e2-组件组合与移除)、[E4 门控配置](experiments.zh-CN.md#e4-bcsg-响应组织) |
| 尺度转换 | 固定采样邻域难以随输入变化 | AAD 常规分支与门控可变形聚合分支并行 | [E3 动态采样替换](experiments.zh-CN.md#e3-aad-动态采样) |
| 跨层上下文 | 浅层高分辨率特征仍需中高层信息 | CIHF 条件交互后在 P2 分辨率聚合 | [E1 有无交互对照](experiments.zh-CN.md#e1-高分辨率与融合结构) |
| 预测端融合 | 不同位置需要的尺度信息不同 | PASF 对每个输出尺度独立对齐并逐位置加权 | [E5 检测头替换](experiments.zh-CN.md#e5-检测头替换) |
| 回归目标 | 区分结构改进和损失替换的作用 | 保留 CIoU、分类损失和 DFL 训练框架 | [E6 回归损失比较](experiments.zh-CN.md#e6-回归损失比较) |

## 总体配置怎样连接

结构入口为 [`yolo11-fpo.yaml`](../../ultralytics/cfg/models/11/yolo11-fpo.yaml)，下表层号按 YAML 从 0 开始计数。通道数会受 n/s/m/l/x 的缩放规则影响；数据 YAML 决定实际类别数。

| 层号 | 输出或操作 | 在完整路径中的位置 |
|---|---|---|
| 0 | 3×3、stride 2 的 Conv | 生成 P1 分辨率的 stem 特征 |
| 1–2、3–4、5–8 | AAD、C3k2_BCSG；末端接 SPPF/C2PSA | 输出 B2、B3、B4；最深层为 P4 |
| 9 | `ScalSeq_SBA([B2, B3, B4])` | CIHF 输出高分辨率聚合特征 |
| 10–12 | 与 B2 拼接、1×1 映射、C3k2_BCSG | 得到 N2 |
| 13–16 | AAD 后与 B3 拼接，再映射与局部建模 | 得到 N3 |
| 17–20 | AAD 后与 B4 拼接，再映射与局部建模 | 得到 N4 |
| 21 | `Detect_PASF_Final([N2, N3, N4])` | 生成 P2/P3/P4 预测 |

对于 s 配置和 640×640 输入，B2/B3/B4 为 128×160×160、256×80×80、256×40×40；N2/N3/N4 为 64×160×160、128×80×80、256×40×40。形状省略 batch 维。[总体图及绘制核对](development.zh-CN.md#图件如何形成)给出对应示意。

## BCSG 从乘法门控到级联残差调制

源码 [`fpo_bcsg.py`](../../ultralytics/nn/modules/fpo_bcsg.py) 的版本说明明确记录了从 BCSG-v1 到 BCSG-v2 的三处改变。这里的 v2 是模块版本，不等于 GitHub 文档第二版。

| 原有处理 | 当前实现 | 设计意图 |
|---|---|---|
| 3×3 depthwise 卷积 | 目标组宽约 32 通道的分组卷积，并寻找可整除分组数 | 在局部建模中保留组内跨通道交互 |
| 特征直接乘空间门控 | `F_sp = F + sigmoid(alpha_sp) * F * g_sp` | 让门控调节增量；门控较低时仍保留基础特征 |
| 两类门控均从原始 F 生成 | 先空间调制，再从 F_sp 估计通道门控并残差相加 | 通道选择能够使用空间调制后的表征 |

两个增量系数的原始参数初始化为 −1.5，经 sigmoid 后约为 0.182。空间响应核以 Scharr-X、Scharr-Y 和均值核初始化，但它们是可学习参数。代码按连续输出区间切分响应，因此图文使用“连续分区”，不将其描述成逐输入通道一一对应的固定方向滤波。

`C3k2_BCSG` 保留分割、串行瓶颈输出收集、拼接和逐点融合的容器方式。其布尔参数控制**内部瓶颈 shortcut**；YAML 中 False 的层仍使用 BCSG。门控配置比较支持当前连续分区方案，但不等于已经独立量化上述三项 v1→v2 修改各自的贡献。

## AAD 在下采样之前聚合可变形响应

源码 [`fpo_aad.py`](../../ultralytics/nn/modules/fpo_aad.py) 中，`AAD_v2` 的两条分支各输出一半目标通道：常规分支使用 stride-2 卷积；动态分支先在原分辨率上执行调制可变形卷积，再平均池化。两路按可学习标量的 sigmoid 加权、拼接，经 1×1 卷积融合。

动态分支 `MDCv3` 的实际算子是 **MMCV `ModulatedDeformConv2d`，即 DCNv2**。其实现分别预测两个偏移分量和调制 mask，并按下式处理偏移：

```text
offset = tanh(predicted_offset) * abs(max_offset) * gate
gate shape = B × 1 × 1 × 1
```

因此幅度门控是每个样本共享的标量，不能写成每个像素独立的门控图。`max_offset` 可学习；偏移幅度受当前参数和 gate 控制。代码定义的 `res_scale` 没有进入当前 `forward` 返回路径，本页不将其作为实际生效的残差分支。源码注释中的“参数减半”“更稳定”等表述不直接作为已验证结论；[E3](experiments.zh-CN.md#e3-aad-动态采样)记录的是整套实现的替换结果。

## CIHF 先交互再高分辨率聚合

源码 [`fpo_cihf.py`](../../ultralytics/nn/modules/fpo_cihf.py) 分成两个阶段：

1. `SBA_Precise` 将 B3、B4 投影至共同维度，由深层描述生成条件变换和混合参数，进行双向信息交换，输出细化的中层特征。
2. `ScalSeq_SBA` 将 B2、细化中层特征和原始 B4 投影、对齐至 P2 尺度。当前 `HyperScaleDynamicFusionAllScale` 从三个对齐特征的全局描述符生成尺度/通道权重，再将聚合结果加回 B2 的投影路径。

代码保留了“第二阶段描述符由单一门控来源改为三尺度来源”的设计注释。当前尺度权重形状为 B×C×3×1×1，另一个调制张量为 B×C×1×1，二者均在空间位置间共享；源码变量名 `spatial_gate` 不改变这个事实。融合尺度权重经 sigmoid 生成，不是沿尺度维归一化为和为 1 的 Softmax。

CIHF 的交互与聚合通道在当前配置中为 128；解析器没有对 `ScalSeq_SBA` 的这个输出值再次施加模型宽度倍率。现有对照量化了无交互聚合与完整 CIHF 的差异，没有在这里编造三尺度描述符改动的独立增益。

## PASF 从尺度加权到最终独立检测头

源码 [`fpo_pasf.py`](../../ultralytics/nn/modules/fpo_pasf.py) 保留 ASFF 的尺度对齐和逐位置尺度选择思路，并将最终实现隔离为 `Detect_PASF_Final`，避免模型名称解析到历史检测头。

| 实现步骤 | 当前代码中的处理 |
|---|---|
| 对齐 | `PureAlign` 先做 1×1 投影，再按目标层使用平均池化、最近邻上采样或保持尺度 |
| 生成权重 | 三个对齐输入相加作为描述，经 16 通道隐藏映射、3 通道输出及 BatchNorm 生成 logits |
| 归一化 | logits 除以可学习温度的绝对值，在三个尺度上做 Softmax；权重逐位置变化、在通道间共享 |
| 初始化 | 产生三个 logits 的卷积权重置零，初始尺度权重均等 |
| 细化 | 融合结果经过卷积、方向注意力和投影，并以固定系数 0.5 与 shortcut 组合 |
| 预测 | P2/P3/P4 各自完成融合，接分类与回归分支，保留 16-bin DFL 解码 |

`beta=0.5` 在当前代码中是 buffer，不是可学习参数。温度使用 `abs()`，当前实现未额外设置正数下界。这些都是当前源码行为；本次文档整理没有改变它们。已有检测头替换实验支持 PASF 作为整体的效果，不单独归因到零初始化、温度或方向注意力。

## 从模块文件到可运行模型

只复制四个模块文件不足以完成集成。模型构建还需要以下连接点：

| 文件 | 负责什么 |
|---|---|
| [`modules/__init__.py`](../../ultralytics/nn/modules/__init__.py) | 导出四个模块入口 |
| [`nn/tasks.py`](../../ultralytics/nn/tasks.py) | 解析 AAD/BCSG 的通道与重复次数，给 CIHF 和 PASF 传入多路输入通道 |
| [`yolo11-fpo.yaml`](../../ultralytics/cfg/models/11/yolo11-fpo.yaml) | 指定模块、连接、shortcut 与模型尺度 |
| [`check_model.py`](../../scripts/check_model.py) | 构建指定尺度并查看结构 |
| [`train.py`](../../scripts/train.py)、[`val.py`](../../scripts/val.py) | 训练与评估入口；数据集特定设置另见复现指南 |

阅读时建议按“YAML → 对应模块的 forward → tasks.py 的参数传递 → 实验对照”的顺序查阅。历史记录中的 ASFM/SBA、Detect/ASFF 和 MDCv3 等名称需要结合当时配置理解；复现公开结构应使用当前 YAML 指定的类，不能仅凭近似名称替换权重。

下一步：[查看各项改进如何被实验检验](experiments.zh-CN.md) · [按当前环境运行](reproduction.zh-CN.md)。
