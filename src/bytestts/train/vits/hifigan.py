# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/vocoder/models/hifigan_generator.py
"""HiFi-GAN generator: the VITS waveform decoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from bytestts.train.vits.commons import LRELU_SLOPE, get_padding


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in (dilation[0], dilation[1], dilation[2])
        )
        self.convs2 = nn.ModuleList(
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in range(3)
        )

    def forward(self, x: Tensor) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in (dilation[0], dilation[1])
        )

    def forward(self, x: Tensor) -> Tensor:
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


class HifiganGenerator(nn.Module):
    """As VITS builds it (vits.py:704): conv_pre and conv_post without weight norm, inference_padding 0."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resblock_type: str,
        resblock_dilation_sizes: Sequence[Sequence[int]],
        resblock_kernel_sizes: Sequence[int],
        upsample_kernel_sizes: Sequence[int],
        upsample_initial_channel: int,
        upsample_factors: Sequence[int],
        cond_channels: int = 0,
        conv_post_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.conv_pre = Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        self.ups = nn.ModuleList(
            weight_norm(
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )
            for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes, strict=True))
        )
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = Conv1d(ch, out_channels, 7, 1, padding=3, bias=conv_post_bias)
        if cond_channels > 0:
            self.cond_layer = nn.Conv1d(cond_channels, upsample_initial_channel, 1)

    def forward(self, x: Tensor, g: Tensor | None = None) -> Tensor:
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = self.ups[i](F.leaky_relu(o, LRELU_SLOPE))
            z_sum = self.resblocks[i * self.num_kernels](o)
            for j in range(1, self.num_kernels):
                z_sum = z_sum + self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)  # default slope 0.01 here, as TTS-dev
        o = self.conv_post(o)
        return torch.tanh(o)
