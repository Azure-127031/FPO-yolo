"""Channel-spatial gating bottleneck (BCSG) and its C3k2 container.

BCSG forms local features, applies spatial residual modulation, and then
computes channel residual modulation from the updated features. Spatial
response kernels are initialized with Scharr and mean filters and remain
learnable. Their outputs are sliced into contiguous channel intervals.

C3k2_BCSG defaults to BCSG blocks. Its trailing c3k option selects standard
C3k blocks without changing the meaning of existing positional arguments."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

def _select_groups(
    channels: int,
    target_group_width: int = 32,
) -> int:
    """Find a channel divisor near the requested group width."""
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
    """Map local responses to a spatial gate with shape (B, 1, H, W).

    The trainable kernel bank starts with Scharr-X, Scharr-Y, and mean-filter
    blocks. Grouped convolution returns 3C channels, split into three contiguous
    ranges. dx, dy, and low name those ranges, rather than paired derivatives
    for every input channel."""

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

        # Initialize contiguous Scharr-X, Scharr-Y, and mean-filter blocks.
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

        # Slice output-channel intervals without per-input-channel regrouping.
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
    """Map a pooled descriptor to a channel gate with shape (B, C, 1, 1).

    A 1D convolution mixes neighboring descriptor entries; a pointwise
    projection mixes all channels. gamma combines their outputs, and
    temperature scales the result before sigmoid."""

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

        # Collapse H and W; keep one descriptor entry per channel.
        desc = self.avg_pool(x)

        # Conv1d reads the channel descriptor as a sequence of length C.
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
    """Refine a feature with spatial gating followed by channel gating.

    Each gate adds a weighted increment to its input. The channel gate reads
    the spatially updated feature. shortcut adds the original bottleneck input
    after both updates."""

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

        self.cv1 = Conv(
            c,
            c,
            1,
            1,
        )

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

        self.intra_spatial = (
            FreqAware_Scharr_Gate(c)
        )

        self.intra_channel = (
            FCAttn_Gate_Plus(c)
        )

        # Sigmoid maps the initial -1.5 strengths to approximately 0.182.
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

        base = self.cv2(
            self.cv1(x)
        )

        g_sp = self.intra_spatial(
            base
        )

        alpha_sp = torch.sigmoid(
            self.alpha_sp
        )

        spatial = (
            base
            + alpha_sp * base * g_sp
        )

        # Channel modulation uses the spatially updated features.
        g_ch = self.intra_channel(
            spatial
        )

        alpha_ch = torch.sigmoid(
            self.alpha_ch
        )

        refined = (
            spatial
            + alpha_ch * spatial * g_ch
        )

        return (
            x + refined
            if self.add
            else refined
        )

class C3k2_BCSG(nn.Module):
    """Split channels, run blocks on the second half, and fuse all outputs.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of internal blocks.
        shortcut: Use a residual shortcut inside each block.
        e: Hidden width as a fraction of c2.
        g: Group count for the optional standard C3k blocks.
        c3k: Use standard C3k blocks instead of BCSG; defaults to False.

    YAML arguments are [c2, shortcut, e, g, c3k]. The parser supplies c1 and n.
    Keeping c3k last preserves the argument order of existing model files."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        g: int = 1,
        c3k: bool = False,
    ):
        super().__init__()

        # The trailing option preserves the existing shortcut argument position.
        self.c3k = bool(c3k)

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

        if self.c3k:
            from ultralytics.nn.modules.block import C3k

            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                BCSGBottleneck_Scharr(self.c, shortcut=shortcut)
                for _ in range(n)
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Keep the first half and append each block output from the second half.
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

        # Keep the first half and append each block output from the second half.
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
