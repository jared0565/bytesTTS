# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/utils/helpers.py and TTS/tts/layers/generic/wavenet.py
"""Small tensor helpers shared by the VITS modules."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    return [item for sublist in pad_shape[::-1] for item in sublist]


def sequence_mask(sequence_length: Tensor, max_len: int | Tensor | None = None) -> Tensor:
    if max_len is None:
        max_len = sequence_length.max()
    # a Tensor max_len stays a Tensor (never .item()), so an ONNX trace keeps the length dynamic
    seq_range = torch.arange(cast(int, max_len), dtype=sequence_length.dtype, device=sequence_length.device)
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


def generate_path(duration: Tensor, mask: Tensor) -> Tensor:
    """duration [b, t_x], mask [b, t_x, t_y] -> hard alignment path [b, t_x, t_y]."""
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = sequence_mask(cum_duration.view(b * t_x), t_y).to(mask.dtype).view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    return path * mask


def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, n_channels: int) -> Tensor:
    """TTS-dev wraps this in torch.jit.script; plain code is the same arithmetic and exports cleanly."""
    in_act = input_a + input_b
    return torch.tanh(in_act[:, :n_channels, :]) * torch.sigmoid(in_act[:, n_channels:, :])
