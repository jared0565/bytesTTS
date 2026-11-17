# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/models/vits.py (Vits.__init__, inference, upsampling_z) and
# TTS/tts/layers/vits/networks.py (PosteriorEncoder)
"""The VITS generator: Coqui's submodule names, so converted checkpoints load with strict=True."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bytestts.train.vits.commons import generate_path, sequence_mask
from bytestts.train.vits.config import VitsModelConfig
from bytestts.train.vits.duration import DurationPredictor, StochasticDurationPredictor
from bytestts.train.vits.flows import ResidualCouplingBlocks
from bytestts.train.vits.hifigan import HifiganGenerator
from bytestts.train.vits.text_encoder import TextEncoder
from bytestts.train.vits.wavenet import WN


class PosteriorEncoder(nn.Module):
    """Spectrogram -> posterior latent. Training only (Phase 4); present so the keys match Coqui's."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: Tensor, x_lengths: Tensor, g: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask


class VitsGenerator(nn.Module):
    def __init__(self, cfg: VitsModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        spk = cfg.embedded_speaker_dim
        if spk:  # vits.py:774, created first as in TTS-dev
            self.emb_g = nn.Embedding(cfg.num_speakers, spk)
        self.text_encoder = TextEncoder(
            cfg.num_chars,
            cfg.hidden_channels,
            cfg.hidden_channels,
            cfg.hidden_channels_ffn_text_encoder,
            cfg.num_heads_text_encoder,
            cfg.num_layers_text_encoder,
            cfg.kernel_size_text_encoder,
            cfg.dropout_p_text_encoder,
        )
        self.posterior_encoder = PosteriorEncoder(
            cfg.out_channels,
            cfg.hidden_channels,
            cfg.hidden_channels,
            kernel_size=cfg.kernel_size_posterior_encoder,
            dilation_rate=cfg.dilation_rate_posterior_encoder,
            num_layers=cfg.num_layers_posterior_encoder,
            cond_channels=spk,
        )
        self.flow = ResidualCouplingBlocks(
            cfg.hidden_channels,
            cfg.hidden_channels,
            kernel_size=cfg.kernel_size_flow,
            dilation_rate=cfg.dilation_rate_flow,
            num_layers=cfg.num_layers_flow,
            cond_channels=spk,
        )
        self.duration_predictor: StochasticDurationPredictor | DurationPredictor
        if cfg.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                cfg.hidden_channels,
                192,
                3,
                cfg.dropout_p_duration_predictor,
                4,
                cond_channels=spk if cfg.condition_dp_on_speaker else 0,
            )
        else:
            self.duration_predictor = DurationPredictor(
                cfg.hidden_channels, 256, 3, cfg.dropout_p_duration_predictor, cond_channels=spk
            )
        self.waveform_decoder = HifiganGenerator(
            cfg.hidden_channels,
            1,
            cfg.resblock_type_decoder,
            cfg.resblock_dilation_sizes_decoder,
            cfg.resblock_kernel_sizes_decoder,
            cfg.upsample_kernel_sizes_decoder,
            cfg.upsample_initial_channel_decoder,
            cfg.upsample_rates_decoder,
            cond_channels=spk,
            conv_post_bias=False,
        )

    def upsampling_z(self, z: Tensor, y_lengths: Tensor, y_mask: Tensor) -> tuple[Tensor, Tensor]:
        """vits.py:944, the inference half: no-op unless the voice decodes at a higher rate than it encodes."""
        factor = self.cfg.interpolate_factor
        if factor and self.cfg.interpolate_z:
            z = F.interpolate(z, scale_factor=[factor], mode="linear").squeeze(0)  # squeeze(0) as TTS-dev
            y_mask = sequence_mask(y_lengths * factor, None).to(y_mask.dtype).unsqueeze(1)
        return z, y_mask

    def infer(
        self,
        tokens: Tensor,
        token_lengths: Tensor,
        *,
        noise_scale: float,
        length_scale: float,
        noise_scale_dp: float,
        sid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """tokens [B, T] -> (waveform [B, 1, S], int64 durations [B, T]), as Vits.inference (vits.py:1089)."""
        with torch.no_grad():  # not the decorator: it is untyped where torch is absent (the gate's mypy)
            return self._infer(tokens, token_lengths, noise_scale, length_scale, noise_scale_dp, sid)

    def _infer(
        self,
        tokens: Tensor,
        token_lengths: Tensor,
        noise_scale: float | Tensor,
        length_scale: float | Tensor,
        noise_scale_dp: float | Tensor,
        sid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """infer's body. The scales may be 0-d tensors, which keeps them live inputs in an ONNX export."""
        g = None
        if hasattr(self, "emb_g"):
            if sid is None:  # TTS-dev would fail later inside the decoder's cond_layer
                raise ValueError("this voice has several speakers: pass a speaker id (sid)")
            g = self.emb_g(sid.unsqueeze(0) if sid.ndim == 0 else sid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(tokens, token_lengths)
        g_dp = g if self.cfg.condition_dp_on_speaker else None
        if isinstance(self.duration_predictor, StochasticDurationPredictor):
            logw = self.duration_predictor(x, x_mask, g=g_dp, reverse=True, noise_scale=noise_scale_dp)
        else:
            logw = self.duration_predictor(x, x_mask, g=g_dp)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))
        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        z, y_mask = self.upsampling_z(z, y_lengths, y_mask)
        o = self.waveform_decoder((z * y_mask)[:, :, : self.cfg.max_inference_len], g=g)
        return o, w_ceil.squeeze(1).long()
