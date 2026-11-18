# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/layers/vits/networks.py (TextEncoder)
"""Token IDs -> prior mean and log-scale per token."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from bytestts.train.vits.commons import sequence_mask
from bytestts.train.vits.transformer import RelativePositionTransformer


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels  # the embedding and its sqrt scale stay this wide
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        hidden_channels += language_emb_dim  # the language vector is concatenated to every token (networks.py:62)
        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: Tensor, x_lengths: Tensor, lang_emb: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """x [B, T] token IDs -> (hidden [B, C + L, T], mean [B, C, T], log-scale [B, C, T], mask [B, 1, T]).

        `lang_emb` [1 or B, L, 1] is the language vector; L is 0 without one."""
        if x.shape[0] != x_lengths.shape[0]:
            raise ValueError("x and x_lengths disagree on the batch size")
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, C]
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)
        x = torch.transpose(x, 1, -1)  # [B, C, T]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask
