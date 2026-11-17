# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/layers/generic/normalization.py
"""Channel-wise layer norms over [B, C, T] tensors."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Layer norm over channels with Coqui's glow-style init (gamma 0.1). Used by DurationPredictor."""

    def __init__(self, channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.gamma + self.beta


class LayerNorm2(nn.Module):
    """torch layer_norm over the channel axis of [B, C, T]."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
