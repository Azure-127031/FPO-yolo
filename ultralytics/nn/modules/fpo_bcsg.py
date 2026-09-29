"""
BCSG-v2: Structure-Guided Residual Enhancement Bottleneck
==========================================================

基于原始 BCSG-v1，仅进行三项核心结构修改：

1. 主体局部卷积：
   原 3×3 Depthwise Conv (g=c)
   -> 固定组宽思想的 3×3 Group Conv
   默认每组约 32 个通道，并自动寻找可整除的 groups。

2. 空间门控作用方式：
   原：
       F <- F * M_sp
   改：
       F_sp = F + alpha_sp * F * g_sp

   空间门控作为“结构增强分支”，不直接压制基础特征。

3. 通道门控顺序：
   原：
       g_sp = SpatialGate(F)
       g_ch = ChannelGate(F)
       F_ref = F * M_sp * M_ch

   改：
       g_sp = SpatialGate(F)
       F_sp = F + alpha_sp * F * g_sp
       g_ch = ChannelGate(F_sp)
       F_ref = F_sp + alpha_ch * F_sp * g_ch

   即：
       局部结构定位 -> 空间残差增强 -> 通道选择 -> 通道残差增强

保留：
- Scharr-X / Scharr-Y / 3×3 Mean 初始化；
- type-major kernel 排列；
- 连续切分 dx / dy / low；
- 可学习 freq_kernel；
- FCAttn_Gate_Plus 的局部/全局通道建模；
- C3k2_BCSG 类名与 YAML 接口；
- 修正后的统一 shortcut 逻辑。

说明：
- alpha_sp / alpha_ch 使用 sigmoid 约束到 (0, 1)；
- 初值设为 -1.5，对应 sigmoid(-1.5) ≈ 0.182，
  使训练初期增强分支较温和，主体特征流接近 identity-preserving。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

def _select_groups(
    channels: int,
    target_group_width: int = 32,
) -> int:
    """
    根据“每组约 target_group_width 个通道”的原则选择 groups，
    并保证 channels % groups == 0。

    例如：
        C=38  -> g=1
        C=64  -> g=2
        C=76  -> g=2
        C=102 -> g=3
        C=128 -> g=4
        C=256 -> g=8
    """
    if channels <= 0:
        raise ValueError(
            f"channels must be positive, got {channels}"
        )

    preferred = max(
        1,
        channels // target_group_width,
    )

    groups = preferred

    while groups > 1:
        if channels % groups == 0:
            break
        groups -= 1

    return groups

class FreqAware_Scharr_Gate(nn.Module):
    """
    原始 BCSG-v1 的分区式频率感知空间门控。

    保留 type-major 核排列：
        [全部 Scharr-X, 全部 Scharr-Y, 全部 Low-pass]

    保留连续切分：
        dx  = freq_feat[:, :C]
        dy  = freq_feat[:, C:2C]
        low = freq_feat[:, 2C:3C]

    注意：
    这里不改你已经验证过的“分区式”实现。
    """

    def __init__(self, dim: int):
        super().__init__()

        if dim <= 0:
            raise ValueError(
                f"dim must be positive, got {dim}"
            )

        self.dim = dim

        scharr_x = torch.tensor(
            [
                [-3.0, 0.0, 3.0],
                [-10.0, 0.0, 10.0],
                [-3.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ) * 0.01

        scharr_y = torch.tensor(
            [
                [-3.0, -10.0, -3.0],
                [0.0, 0.0, 0.0],
                [3.0, 10.0, 3.0],
            ],
            dtype=torch.float32,
        ) * 0.01

        low = torch.ones(
            (3, 3),
            dtype=torch.float32,
        ) / 9.0

        # type-major:
        # [X1...XC, Y1...YC, L1...LC]
        freq_kernel = torch.zeros(
            dim * 3,
            1,
            3,
            3,
            dtype=torch.float32,
        )

        freq_kernel[:dim, 0] = (
            scharr_x.unsqueeze(0)
            .repeat(dim, 1, 1)
        )

        freq_kernel[dim:2 * dim, 0] = (
            scharr_y.unsqueeze(0)
            .repeat(dim, 1, 1)
        )

        freq_kernel[2 * dim:, 0] = (
            low.unsqueeze(0)
            .repeat(dim, 1, 1)
        )

        # 保留原始“可学习频率核”设定
        self.freq_kernel = nn.Parameter(
            freq_kernel,
            requires_grad=True,
        )

        hidden = max(
            dim // 16,
            4,
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                dim * 2,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden,
                1,
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                f"Expected BCHW input, "
                f"got shape {tuple(x.shape)}"
            )

        _, c, _, _ = x.shape

        if c != self.dim:
            raise ValueError(
                f"Input channels ({c}) do not "
                f"match initialized dim ({self.dim})"
            )

        kernel = self.freq_kernel.to(
            device=x.device,
            dtype=x.dtype,
        )

        freq_feat = F.conv2d(
            x,
            kernel,
            bias=None,
            stride=1,
            padding=1,
            groups=c,
        )

        # 保留原始连续区间切分
        dx = freq_feat[:, :c]
        dy = freq_feat[:, c:2 * c]
        low = freq_feat[:, 2 * c:]

        edge_raw = dx + dy

        gate_input = torch.cat(
            (edge_raw, low),
            dim=1,
        )

        g_sp = self.fuse(
            gate_input
        )

        return g_sp

class FCAttn_Gate_Plus(nn.Module):
    """
    原始 BCSG-v1 的局部-全局通道门控。

    输入：
        F_sp

    流程：
        GAP
          ├─ Local 1D Conv
          └─ Global 1×1 Conv
        -> gamma 融合
        -> temperature 缩放
        -> Sigmoid
        -> g_ch

    与 v1 相比，结构本身不改。
    唯一区别是 v2 中它接收空间增强后的 F_sp，
    而不是原始 base F。
    """

    def __init__(
        self,
        channel: int,
        b: int = 1,
        gamma: int = 2,
    ):
        super().__init__()

        if channel <= 0:
            raise ValueError(
                f"channel must be positive, "
                f"got {channel}"
            )

        t = int(
            abs(
                (
                    math.log(channel, 2)
                    + b
                ) / gamma
            )
        )

        k = (
            t
            if t % 2
            else t + 1
        )

        k = max(
            k,
            1,
        )

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.conv1d = nn.Conv1d(
            1,
            1,
            kernel_size=k,
            padding=k // 2,
            bias=False,
        )

        self.fc = nn.Conv2d(
            channel,
            channel,
            kernel_size=1,
            bias=False,
        )

        self.gamma = nn.Parameter(
            torch.tensor(0.5)
        )

        self.temperature = nn.Parameter(
            torch.tensor(1.0)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        desc = self.avg_pool(x)

        local = (
            self.conv1d(
                desc
                .squeeze(-1)
                .transpose(-1, -2)
            )
            .transpose(-1, -2)
            .unsqueeze(-1)
        )

        global_ = self.fc(
            desc
        )

        combined = (
            self.gamma * local
            + (1.0 - self.gamma) * global_
        ) * self.temperature

        g_ch = self.sigmoid(
            combined
        )

        return g_ch

class BCSGBottleneck_Scharr(nn.Module):
    """
    BCSG-v2 Bottleneck

    核心路径：

        X
        |
        v
      1×1 Conv
        |
        v
      3×3 GroupConv
        |
        v
      base F
        |
        +------> Spatial Gate ------> g_sp
        |                             |
        +-----------------------------+
        |
        v
      F_sp = F + alpha_sp * F * g_sp
        |
        +------> Channel Gate ------> g_ch
        |                             |
        +-----------------------------+
        |
        v
      F_ref = F_sp + alpha_ch * F_sp * g_ch
        |
        v
      + shortcut (optional)

    设计目的：
    - 主体 GroupConv 恢复局部跨通道交互；
    - 空间门控只做结构增强，不直接删除弱响应；
    - 通道选择基于已经完成空间增强的特征；
    - 两级增强均采用 residual form，保持稳定特征流。
    """

    def __init__(
        self,
        c: int,
        shortcut: bool = True,
        target_group_width: int = 32,
    ):
        super().__init__()

        if c <= 0:
            raise ValueError(
                f"c must be positive, got {c}"
            )

        self.c = c

        # 第一层保持原设计
        self.cv1 = Conv(
            c,
            c,
            1,
            1,
        )

        # v2 修改 1：
        # Depthwise(g=c) -> GroupConv
        self.groups = _select_groups(
            c,
            target_group_width,
        )

        self.cv2 = Conv(
            c,
            c,
            3,
            1,
            g=self.groups,
        )

        # 空间结构门控
        self.intra_spatial = (
            FreqAware_Scharr_Gate(c)
        )

        # 局部-全局通道门控
        self.intra_channel = (
            FCAttn_Gate_Plus(c)
        )

        # v2 修改 2/3：
        # 两个 enhancement strength 均采用受约束的可学习标量。
        #
        # sigmoid(-1.5) ≈ 0.182
        # 训练初期只进行温和增强。
        self.alpha_sp = nn.Parameter(
            torch.tensor(-1.5)
        )

        self.alpha_ch = nn.Parameter(
            torch.tensor(-1.5)
        )

        self.add = bool(
            shortcut
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # -------------------------
        # 1. 基础局部表征
        # -------------------------
        base = self.cv2(
            self.cv1(x)
        )

        # -------------------------
        # 2. 空间结构选择
        # -------------------------
        g_sp = self.intra_spatial(
            base
        )

        alpha_sp = torch.sigmoid(
            self.alpha_sp
        )

        # 残差式空间增强：
        # gate 低 -> 保留 base
        # gate 高 -> 增强对应位置
        spatial = (
            base
            + alpha_sp * base * g_sp
        )

        # -------------------------
        # 3. 通道选择
        # -------------------------
        # 关键：
        # channel gate 不再直接从 base 生成，
        # 而是在空间结构增强之后生成。
        g_ch = self.intra_channel(
            spatial
        )

        alpha_ch = torch.sigmoid(
            self.alpha_ch
        )

        # 残差式通道增强
        refined = (
            spatial
            + alpha_ch * spatial * g_ch
        )

        # -------------------------
        # 4. Bottleneck shortcut
        # -------------------------
        return (
            x + refined
            if self.add
            else refined
        )

class C3k2_BCSG(nn.Module):
    """
    BCSG-v2 的 C3k2 容器。

    YAML 接口保持兼容：
        C3k2_BCSG, [c2, shortcut, e]

    shortcut 使用修正后的统一语义：
    - shortcut=True:
        所有内部 BCSG Bottleneck 均使用 shortcut
    - shortcut=False:
        所有内部 BCSG Bottleneck 均关闭 shortcut

    对 YOLO11s 而言 depth scaling 后通常 n=1，
    对更大 scale 也保持一致、清晰的语义。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        g: int = 1,
    ):
        super().__init__()

        # 保留接口兼容
        _ = g

        self.c = int(
            c2 * e
        )

        if self.c <= 0:
            raise ValueError(
                f"hidden channels must be positive, "
                f"got c2={c2}, e={e}, hidden={self.c}"
            )

        self.cv1 = Conv(
            c1,
            2 * self.c,
            1,
            1,
        )

        self.cv2 = Conv(
            (2 + n) * self.c,
            c2,
            1,
            1,
        )

        self.m = nn.ModuleList(
            BCSGBottleneck_Scharr(
                self.c,
                shortcut=shortcut,
            )
            for _ in range(n)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y = list(
            self.cv1(x).chunk(
                2,
                dim=1,
            )
        )

        for block in self.m:
            y.append(
                block(y[-1])
            )

        return self.cv2(
            torch.cat(
                y,
                dim=1,
            )
        )

    def forward_split(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y = list(
            self.cv1(x).split(
                (self.c, self.c),
                dim=1,
            )
        )

        for block in self.m:
            y.append(
                block(y[-1])
            )

        return self.cv2(
            torch.cat(
                y,
                dim=1,
            )
        )
