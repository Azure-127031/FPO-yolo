# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Cross-level interaction and high-resolution fusion (CIHF)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def Upsample(x, size, align_corners=False):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        pad = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, pad, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SBA_Precise(nn.Module):
    def __init__(self, cH, cL, mid_dim=128):
        super().__init__()
        self.mid_dim = mid_dim
        self.fc_l = nn.Sequential(
            nn.Conv2d(cL, cL, 3, 1, 1, groups=cL, bias=False),
            nn.Conv2d(cL, mid_dim, 1, bias=False),
            nn.GroupNorm(4, mid_dim),
        )
        self.fc_h = nn.Sequential(
            nn.Conv2d(cH, cH, 3, 1, 1, groups=cH, bias=False),
            nn.Conv2d(cH, mid_dim, 1, bias=False),
            nn.GroupNorm(4, mid_dim),
        )
        self.gate_to_proc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_dim, mid_dim * 3 + 3, 1, bias=False),
            nn.GroupNorm(3, mid_dim * 3 + 3),
            nn.Sigmoid(),
        )
        self.fuse = Conv(mid_dim * 2, mid_dim, 1)

    def forward(self, H, L):
        L_mid = self.fc_l(L)
        H_mid = self.fc_h(H)
        g_L = torch.sigmoid(L_mid)
        g_H = torch.sigmoid(H_mid)
        proc_params = self.gate_to_proc(F.adaptive_avg_pool2d(g_H, 1))
        mid = self.mid_dim
        proc_l_w = proc_params[:, :mid]
        proc_l_b = proc_params[:, mid:mid * 2]
        proc_h_w = proc_params[:, mid * 2:mid * 3]
        residue = proc_params[:, mid * 3:]
        proc_h_b = residue[:, :1]
        alpha_l = residue[:, 1:2]
        alpha_h = residue[:, 2:3]
        L_proc = L_mid * proc_l_w + proc_l_b
        H_proc = H_mid * proc_h_w + proc_h_b
        L_proc_gated = L_proc * g_L
        H_proc_gated = H_proc * g_H
        H_up = Upsample(H_proc_gated, L_mid.shape[2:])
        L_down = Upsample(L_proc_gated, H_mid.shape[2:])
        alpha_l = Upsample(alpha_l, L_mid.shape[-2:])
        alpha_h = Upsample(alpha_h, H_mid.shape[-2:])
        L_out = L_mid + alpha_l * L_proc_gated + (1.0 - alpha_l) * H_up
        H_out = H_mid + alpha_h * H_proc_gated + (1.0 - alpha_h) * L_down
        H_out_up = Upsample(H_out, L_out.shape[2:])
        return self.fuse(torch.cat([L_out, H_out_up], dim=1)), g_L, g_H

class HyperScaleDynamicFusionAllScale(nn.Module):
    """
    CIHF ablation 3: Stage-2 gating descriptor uses all three aligned
    P2/P3/P4 features instead of only pooled g_L.

    The gate is still channel-wise and spatial-shared; only its conditioning
    source changes.
    """

    def __init__(self, channels, num_scales=3):
        super().__init__()
        desc_channels = channels * num_scales
        self.hypernet = nn.Sequential(
            nn.Conv2d(desc_channels, channels * num_scales, 1, bias=False),
            nn.GroupNorm(8, channels * num_scales),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(desc_channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x_scales, g_L=None, g_H=None):
        B, C, S, H, W = x_scales.shape
        pooled = [F.adaptive_avg_pool2d(x_scales[:, :, s], 1) for s in range(S)]
        descriptor = torch.cat(pooled, dim=1)
        scale_weights = self.hypernet(descriptor).view(B, C, S, 1, 1)
        x_fused = (x_scales * scale_weights).sum(dim=2)
        spatial_mod = self.spatial_gate(descriptor).view(B, C, 1, 1)
        return self.bn(self.spatial_conv(x_fused * spatial_mod))

class ScalSeq_SBA(nn.Module):
    """CIHF ablation: all-scale descriptor for Stage-2 gate generation."""

    def __init__(self, inc, out_ch=128, sba_dim=128):
        super().__init__()
        c2, c3, c4 = inc
        self.sba = SBA_Precise(cH=c4, cL=c3, mid_dim=sba_dim)
        self.conv_p2 = Conv(c2, out_ch, 1)
        self.conv_p3 = Conv(sba_dim, out_ch, 1)
        self.conv_p4 = Conv(c4, out_ch, 1)
        self.fusion = HyperScaleDynamicFusionAllScale(out_ch, num_scales=3)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.act = nn.SiLU()

    def forward(self, x):
        p2, p3, p4 = x
        p2_base = self.conv_p2(p2)
        p3_sba, g_L, g_H = self.sba(p4, p3)
        p3_c = self.conv_p3(p3_sba)
        p4_c = self.conv_p4(p4)
        size = p2_base.shape[2:]
        p3_c = Upsample(p3_c, size)
        p4_c = Upsample(p4_c, size)
        x_scales = torch.stack([p2_base, p3_c, p4_c], dim=2)
        y_fused = self.fusion(x_scales, g_L, g_H)
        out = p2_base + self.beta * y_fused
        return self.act(out)
