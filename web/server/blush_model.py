"""RELION's Blush network, verbatim from relion_blush/model/model.py
(https://github.com/3dem/relion-blush, MIT License, Copyright (c) 2023 3dem),
so that the published checkpoint's state dictionary loads without any
renaming. The checkpoint itself (blush_v1.0.ckpt.gz, Zenodo record 10072731)
also carries this definition as text; blush.py checks the two agree. Only the
module class is used here; the hyper-parameters below are the ones the v1.0
weights were trained with."""

import torch
import torch.nn as nn
from collections import OrderedDict

BLOCK_SIZE = 64
DEPTH = 5
WIDTH = 16
TRILINEAR = True
MASK_INFER = True

NORM_MODULE = nn.InstanceNorm3d  # e.g. nn.InstanceNorm3d or nn.BatchNorm3d
ACTIVATION = nn.SiLU(True)  # e.g. nn.SiLU or nn.ReLU


class DoubleConv(nn.Module):
    """(convolution => [BN] => activation) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            OrderedDict(
                [
                    ('conv1', nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)),
                    ('norm1', NORM_MODULE(mid_channels, affine=True)),
                    ('act1', ACTIVATION),
                    ('conv2', nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)),
                    ('norm2', NORM_MODULE(out_channels, affine=True)),
                    ('act2', ACTIVATION)
                ]
            )
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, pad=False):
        super().__init__()
        self.pad = pad
        if trilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                mid_channels=in_channels // 2
            )
        else:
            self.up = torch.nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.pad:
            diffZ = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            diffX = x2.size()[4] - x1.size()[4]
            x1 = torch.nn.functional.pad(
                x1,
                [
                    diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2,
                    diffZ // 2, diffZ - diffZ // 2
                ]
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BlushModel(torch.nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(BlushModel, self).__init__()
        factor = 2 if TRILINEAR else 1
        self.down = []
        self.up = []
        self.inc = DoubleConv(
            in_channels=in_channels,
            out_channels=WIDTH
        )
        self.down_pool = torch.nn.MaxPool3d(2)
        for i in range(DEPTH - 1):
            n = 2 ** i
            self.down.append(
                DoubleConv(
                    in_channels=WIDTH * n,
                    out_channels=WIDTH * n * 2
                )
            )
        self.down.append(
            DoubleConv(
                in_channels=WIDTH * 2 ** (DEPTH - 1),
                out_channels=WIDTH * 2 ** DEPTH // factor
            )
        )
        self.down = torch.nn.ModuleList(self.down)
        for i in range(DEPTH - 1):
            n = 2 ** (DEPTH - 1 - i)
            self.up.append(
                Up(
                    in_channels=WIDTH * n * 2,
                    out_channels=WIDTH * n // factor,
                    trilinear=TRILINEAR
                )
            )
        self.up.append(
            Up(
                in_channels=WIDTH * 2,
                out_channels=WIDTH,
                trilinear=TRILINEAR
            )
        )
        self.up = torch.nn.ModuleList(self.up)
        self.outc = torch.nn.Conv3d(
            in_channels=WIDTH,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, grid, local_std):
        std = torch.std(grid, (-1, -2, -3), keepdim=True)
        mean = torch.mean(grid, (-1, -2, -3), keepdim=True)
        grid_standard = (grid - mean) / (std + 1e-12)
        input = torch.cat([grid_standard.unsqueeze(1), local_std.unsqueeze(1)], 1)
        nn = self.inc(input)
        skip = []
        for i in range(DEPTH):
            skip.append(nn)
            nn = self.down_pool(nn)
            nn = self.down[i](nn)
        skip = skip[::-1]
        for i in range(DEPTH):
            nn = self.up[i](nn, skip[i])
        nn = self.outc(nn)
        output = grid_standard - nn[:, 0]
        output = output * (std + 1e-12) + mean
        mask_logit = nn[:, 1] if MASK_INFER else None
        return output, mask_logit
