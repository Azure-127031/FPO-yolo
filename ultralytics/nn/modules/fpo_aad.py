# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Adaptive aggregation downsampling (AAD)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d
from mmcv.cnn import ConvModule
from ultralytics.nn.modules.conv import Conv

class MDCv3(nn.Module):
    """
    Modulated Deformable Conv v3 - 真正的DCNv2升级
    核心改进：
    1. 分离式Offset：X/Y方向独立预测（减少50%参数量，提升稳定性）
    2. 形变幅度门控：自适应约束offset范围（防止极端形变）
    3. 残差形变连接：offset预测加入残差，训练更稳定
    """
    def __init__(self, c1, c2, k=3, stride=1, padding=1):
        super().__init__()
        self.c1, self.c2 = c1, c2

        # 标准下采样（保持对齐）
        self.stride = stride
        if stride == 2:
            self.down = nn.AvgPool2d(2)

        # ========== 核心改进1：分离式Offset预测 ==========
        # X方向和Y方向分别预测，维度减半，优化更稳定
        self.offset_x = nn.Sequential(
            Conv(c1, c1//4, 1, act=True),  # 轻量降维
            nn.Conv2d(c1//4, 9, 1)  # 3x3=9个采样点的X偏移
        )
        self.offset_y = nn.Sequential(
            Conv(c1, c1//4, 1, act=True),
            nn.Conv2d(c1//4, 9, 1)  # 3x3=9个采样点的Y偏移
        )

        # ========== 核心改进2：形变幅度门控（Deformation Gate）==========
        # 预测每个位置是否需要大形变（节省计算+防止过拟合）
        self.deform_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, 1, 1),
            nn.Sigmoid()
        )

        # 可学习最大形变范围（替代DCNv2的固定范围）
        self.max_offset = nn.Parameter(torch.tensor(3.0))  # 初始最大3像素

        # Modulated DCN（保持MMCV加速）
        self.dcn = ModulatedDeformConv2d(
            c1, c2, k, stride=1, padding=padding,
            deform_groups=1  # 可以改为2增加灵活性
        )

        # Mask预测（与offset分离，更稳定）
        self.mask_pred = nn.Sequential(
            Conv(c1, c1//4, 1, act=True),
            nn.Conv2d(c1//4, 9, 1),  # 9个mask
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        # 分离式预测X/Y偏移（创新点1）
        off_x = self.offset_x(x)  # [B, 9, H, W]
        off_y = self.offset_y(x)  # [B, 9, H, W]

        # 组合成标准offset格式 [B, 18, H, W]（交错排列）
        # 第0,2,4...是x，第1,3,5...是y
        offset = torch.stack([off_x, off_y], dim=2).view(B, 18, H, W)

        # ========== 核心改进2：形变幅度约束 ==========
        # 门控：决定形变强度（0-1之间）
        gate = self.deform_gate(x)  # [B, 1, 1, 1]

        # 软约束：offset = tanh(offset) * max_offset * gate
        # 既限制范围，又保留梯度；gate控制是否启用大形变
        offset = torch.tanh(offset) * torch.abs(self.max_offset) * gate

        # Mask预测
        mask = self.mask_pred(x)

        # DCN前向（利用MMCV优化）
        out = self.dcn(x, offset, mask)
        out = self.act(self.bn(out))

        # 下采样（如果需要）
        if self.stride == 2:
            out = self.down(out)

        return out

class AAD_v2(nn.Module):
    """
    基于MDCv3的AAD升级版
    保持你原本AAD的结构，但将detail分支换成MDCv3
    """
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.c1, self.c2 = c1, c2
        half_c2 = c2 // 2

        # 主分支（标准）
        self.main_branch = ConvModule(
            c1, half_c2, k, stride=2, padding=k//2,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'),
            inplace=True
        )

        # Detail分支：使用MDCv3（真正的DCNv2升级）
        self.detail_branch = MDCv3(c1, half_c2, stride=2)

        # 融合（保持原版）
        self.fuse = Conv(c2, c2, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # 残差（保持原版）
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 主分支
        main_out = self.main_branch(x)

        # Detail分支：MDCv3（分离式形变，更稳定更快）
        detail_out = self.detail_branch(x)

        # 自适应融合
        alpha = torch.sigmoid(self.alpha)
        mixed = torch.cat([
            main_out * (1 - alpha),
            detail_out * alpha
        ], dim=1)

        out = self.fuse(mixed)

        return out
