# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Adaptive aggregation downsampling (AAD): regular and deformable branches."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d
from mmcv.cnn import ConvModule
from ultralytics.nn.modules.conv import Conv

class MDCv3(nn.Module):
    """Aggregate features with modulated DCNv2, followed by optional average pooling.

    MDCv3 is the retained implementation name, not the DCNv3 operator.
    Two predictors supply offset components; tanh, a learned magnitude, and
    a sample-wise gate scale them before deformable convolution. A separate
    sigmoid mask modulates the sampled features. Batch normalization and SiLU
    precede pooling. MMCV reads offsets as interleaved height/width values;
    historical offset_x/offset_y keys are preserved in that slot order."""
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

        # MMCV consumes height/width pairs; retain historical parameter names.
        offset = torch.stack([off_x, off_y], dim=2).view(B, 18, H, W)

        gate = self.deform_gate(x)

        offset = torch.tanh(offset) * torch.abs(self.max_offset) * gate

        mask = self.mask_pred(x)

        out = self.dcn(x, offset, mask)
        out = self.act(self.bn(out))

        if self.stride == 2:
            out = self.down(out)

        return out

class AAD_v2(nn.Module):
    """Combine regular stride-2 convolution and deformable aggregation with pooling.

    A learned sigmoid scalar weights the two branches before channel
    concatenation and pointwise fusion. It is shared across inputs after
    training, unlike the adaptive branch's input-dependent amplitude gate."""
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

        # Weight each branch before concatenation, then project the result.
        alpha = torch.sigmoid(self.alpha)
        mixed = torch.cat([
            main_out * (1 - alpha),
            detail_out * alpha
        ], dim=1)

        out = self.fuse(mixed)

        return out
