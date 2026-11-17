# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/layers/generic/wavenet.py
"""Non-causal WaveNet stack (WN) used by the posterior encoder and the coupling flows."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from bytestts.train.vits.commons import fused_add_tanh_sigmoid_multiply


class WN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        c_in_channels: int = 0,
        dropout_p: float = 0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1 or hidden_channels % 2 != 0:
            raise ValueError("WN needs an odd kernel_size and an even hidden_channels")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.c_in_channels = c_in_channels
        self.dropout_p = dropout_p
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_p)
        if c_in_channels > 0:
            self.cond_layer = weight_norm(nn.Conv1d(c_in_channels, 2 * hidden_channels * num_layers, 1), name="weight")
        for i in range(num_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                in_channels if i == 0 else hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.in_layers.append(weight_norm(in_layer, name="weight"))
            res_skip_channels = 2 * hidden_channels if i < num_layers - 1 else hidden_channels
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1), name="weight"))

    def forward(self, x: Tensor, x_mask: Tensor | None = None, g: Tensor | None = None) -> Tensor:
        output = torch.zeros_like(x)
        mask: Tensor | float = 1.0 if x_mask is None else x_mask
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.num_layers):
            x_in = self.dropout(self.in_layers[i](x))
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * mask
