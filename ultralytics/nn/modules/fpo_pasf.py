# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Paper-final PASF head used by FPO-YOLO.

This module is intentionally isolated from the historical ASFF experiments so
the final paper architecture cannot silently resolve to Detect_ASFF1/ASFFV5.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.tal import dist2bbox, make_anchors

from .head import Detect

__all__ = ["Detect_PASF_Final"]


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CoordAttn(nn.Module):
    def __init__(self, c, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden = max(8, c // reduction)
        self.conv = nn.Sequential(Conv(c, hidden, 1, act=True), nn.Conv2d(hidden, c, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = self.pool_h(x).permute(0, 1, 3, 2)
        x_w = self.pool_w(x)
        y = self.conv(torch.cat([x_h, x_w], dim=3))
        a_h, a_w = torch.split(y, [h, w], dim=3)
        a_h = self.sigmoid(a_h.permute(0, 1, 3, 2))
        a_w = self.sigmoid(a_w)
        return x * a_h * a_w


class PureAlign(nn.Module):
    def __init__(self, c1, c2, scale=2, mode="down"):
        super().__init__()
        self.mode = mode
        self.proj = Conv(c1, c2, 1, 1)
        if mode == "down":
            self.pool = nn.AvgPool2d(scale, scale)
        elif mode == "up":
            self.up = nn.Upsample(scale_factor=scale, mode="nearest")
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.mode == "down":
            return self.pool(x)
        if self.mode == "up":
            return self.up(x)
        return x


class ASFF_Stable(nn.Module):
    def __init__(self, level, ch, multiplier=1):
        super().__init__()
        self.level = level
        c2, c3, c4 = ch[0], ch[1], ch[2]

        base_dim = ch[2 - level] // 2
        if level == 0:
            self.inter_dim = min(base_dim, 256)
        elif level == 1:
            self.inter_dim = min(base_dim, 128)
        else:
            self.inter_dim = base_dim

        self.output_dim = int(ch[2 - level] * multiplier)
        if level == 0:
            self.resample_1 = PureAlign(int(c3 * multiplier), self.inter_dim, 2, "down")
            self.resample_2 = PureAlign(int(c2 * multiplier), self.inter_dim, 4, "down")
            self.resample_0 = PureAlign(int(c4 * multiplier), self.inter_dim, 1, "stay")
        elif level == 1:
            self.resample_0 = PureAlign(int(c4 * multiplier), self.inter_dim, 2, "up")
            self.resample_2 = PureAlign(int(c2 * multiplier), self.inter_dim, 2, "down")
            self.resample_1 = PureAlign(int(c3 * multiplier), self.inter_dim, 1, "stay")
        else:
            self.resample_0 = PureAlign(int(c4 * multiplier), self.inter_dim, 4, "up")
            self.resample_1 = PureAlign(int(c3 * multiplier), self.inter_dim, 2, "up")
            self.resample_2 = PureAlign(int(c2 * multiplier), self.inter_dim, 1, "stay")

        self.weight_gen = nn.Sequential(
            Conv(self.inter_dim, 16, 1, act=True),
            nn.Conv2d(16, 3, 1, bias=False),
            nn.BatchNorm2d(3),
        )
        nn.init.constant_(self.weight_gen[1].weight, 0.0)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.expand = nn.Sequential(
            Conv(self.inter_dim, self.inter_dim, 3, 1),
            CoordAttn(self.inter_dim, reduction=32),
            Conv(self.inter_dim, self.output_dim, 1),
        )
        self.shortcut = (
            nn.Identity() if self.inter_dim == self.output_dim else Conv(self.inter_dim, self.output_dim, 1)
        )
        self.register_buffer("beta", torch.tensor(0.5))

    def forward(self, x):
        x2, x3, x4 = x[0], x[1], x[2]
        l0 = self.resample_0(x4)
        l1 = self.resample_1(x3)
        l2 = self.resample_2(x2)
        w = F.softmax(self.weight_gen(l0 + l1 + l2) / self.temp.abs(), dim=1)
        fused = l0 * w[:, 0:1] + l1 * w[:, 1:2] + l2 * w[:, 2:3]
        return self.shortcut(fused) + self.beta * self.expand(fused)


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect_PASF_Final(Detect):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=(), multiplier=1):
        # Bypass Detect.__init__ because PASF defines its own fusion and prediction branches.
        nn.Module.__init__(self)
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.l0_fusion = ASFF_Stable(level=0, ch=ch, multiplier=multiplier)
        self.l1_fusion = ASFF_Stable(level=1, ch=ch, multiplier=multiplier)
        self.l2_fusion = ASFF_Stable(level=2, ch=ch, multiplier=multiplier)

    def forward(self, x):
        x = [self.l2_fusion(x), self.l1_fusion(x), self.l0_fusion(x)]
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            prob = 5 / self.nc / (640 / s) ** 2
            b[-1].bias.data[: self.nc] = math.log(prob)
