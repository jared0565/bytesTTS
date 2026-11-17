# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/layers/vits/networks.py (ResidualCouplingBlock(s))
"""The normalizing flow between the prior and the posterior latent."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from bytestts.train.vits.wavenet import WN


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        dropout_p: float = 0,
        cond_channels: int = 0,
        mean_only: bool = False,
    ) -> None:
        if channels % 2 != 0:
            raise ValueError("channels should be divisible by 2")
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        if self.post.bias is not None:
            self.post.bias.data.zero_()

    def forward(
        self, x: Tensor, x_mask: Tensor, g: Tensor | None = None, reverse: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            return torch.cat([x0, x1], 1), torch.sum(log_scale, [1, 2])
        x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
        return torch.cat([x0, x1], 1)


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows: int = 4,
        cond_channels: int = 0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels
        self.flows = nn.ModuleList(
            ResidualCouplingBlock(
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                num_layers,
                cond_channels=cond_channels,
                mean_only=True,
            )
            for _ in range(num_flows)
        )

    def forward(self, x: Tensor, x_mask: Tensor, g: Tensor | None = None, reverse: bool = False) -> Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
