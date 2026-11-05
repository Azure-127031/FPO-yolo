# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Adaptive aggregation downsampling (AAD): regular and deformable branches."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d
from mmcv.cnn import ConvModule
from ultralytics.nn.modules.conv import Conv

class MDCv3(nn.Module):
    """Predict offsets and a mask, apply DCNv2, then pool when stride is 2.

    The class name is historical; the operator is MMCV ModulatedDeformConv2d.
    offset_x and offset_y supply its height and width slots, respectively.
    Their parameter names are kept so existing checkpoints still load."""
    def __init__(self, c1, c2, k=3, stride=1, padding=1):
        super().__init__()
        self.c1, self.c2 = c1, c2

        self.stride = stride
        if stride == 2:
            self.down = nn.AvgPool2d(2)

        self.offset_x = nn.Sequential(
            Conv(c1, c1//4, 1, act=True),
            nn.Conv2d(c1//4, 9, 1)
        )
        self.offset_y = nn.Sequential(
            Conv(c1, c1//4, 1, act=True),
            nn.Conv2d(c1//4, 9, 1)
        )

        # Global pooling makes the amplitude gate one scalar per sample.
        self.deform_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, 1, 1),
            nn.Sigmoid()
        )

        self.max_offset = nn.Parameter(torch.tensor(3.0))

        self.dcn = ModulatedDeformConv2d(
            c1, c2, k, stride=1, padding=padding,
            deform_groups=1
        )

        self.mask_pred = nn.Sequential(
            Conv(c1, c1//4, 1, act=True),
            nn.Conv2d(c1//4, 9, 1),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        off_x = self.offset_x(x)
        off_y = self.offset_y(x)

        # Interleave the two 9-channel maps as (h1, w1, ..., h9, w9).
        offset = torch.stack([off_x, off_y], dim=2).view(B, 18, H, W)

        # One amplitude gate per image, broadcast to every sampling position.
        gate = self.deform_gate(x)

        # Bound the raw offsets before applying the learned magnitude.
        offset = torch.tanh(offset) * torch.abs(self.max_offset) * gate

        mask = self.mask_pred(x)

        out = self.dcn(x, offset, mask)
        out = self.act(self.bn(out))

        # Aggregate at the input resolution before reducing it by half.
        if self.stride == 2:
            out = self.down(out)

        return out

class AAD_v2(nn.Module):
    """Downsample through a regular branch and a deformable branch.

    Each branch produces half the output channels. A shared learned scalar
    weights them before concatenation and 1x1 fusion. The deformable branch
    has its own input-dependent offset gate."""
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.c1, self.c2 = c1, c2
        half_c2 = c2 // 2

        self.main_branch = ConvModule(
            c1, half_c2, k, stride=2, padding=k//2,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'),
            inplace=True
        )

        self.detail_branch = MDCv3(c1, half_c2, stride=2)

        self.fuse = Conv(c2, c2, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # Retained checkpoint parameter; it is not used by forward.
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.shape

        main_out = self.main_branch(x)

        detail_out = self.detail_branch(x)

        # alpha is shared across inputs; the branches are concatenated, not summed.
        alpha = torch.sigmoid(self.alpha)
        mixed = torch.cat([
            main_out * (1 - alpha),
            detail_out * alpha
        ], dim=1)

        out = self.fuse(mixed)

        return out
