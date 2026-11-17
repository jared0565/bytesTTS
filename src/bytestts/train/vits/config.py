# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/models/vits.py (VitsArgs, lines 366-600)
"""The VitsArgs fields the generator reads, with Coqui's names and defaults."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bytestts.errors import NotSupported

# Conditioning TTS-dev feeds from external managers or files; the port does not carry it.
_REFUSED = {
    "use_d_vector_file": "d-vector (external speaker embedding) voices",
    "use_language_embedding": "language-embedding (multilingual) voices",
}


def _ints(v: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(i) for i in v)


@dataclass(frozen=True)
class VitsModelConfig:
    num_chars: int
    out_channels: int = 513
    spec_segment_size: int = 32
    hidden_channels: int = 192
    hidden_channels_ffn_text_encoder: int = 768
    num_heads_text_encoder: int = 2
    num_layers_text_encoder: int = 6
    kernel_size_text_encoder: int = 3
    dropout_p_text_encoder: float = 0.1
    dropout_p_duration_predictor: float = 0.5
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes_decoder: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates_decoder: tuple[int, ...] = (8, 8, 2, 2)
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: tuple[int, ...] = (16, 16, 4, 4)
    use_sdp: bool = True
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    length_scale: float = 1
    noise_scale_dp: float = 1.0
    inference_noise_scale_dp: float = 1.0
    max_inference_len: int | None = None
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speaker_embedding_channels: int = 256
    condition_dp_on_speaker: bool = True
    encoder_sample_rate: int | None = None
    interpolate_z: bool = True
    sample_rate: int | None = None  # the voice's audio sample rate; needed only with encoder_sample_rate

    def __post_init__(self) -> None:
        if self.resblock_type_decoder not in ("1", "2"):
            raise ValueError(f"resblock_type_decoder must be '1' or '2', not {self.resblock_type_decoder!r}")
        if self.encoder_sample_rate and not self.sample_rate:
            raise ValueError("encoder_sample_rate needs the voice's audio sample_rate")

    @property
    def embedded_speaker_dim(self) -> int:  # vits.py:774
        return self.speaker_embedding_channels if self.use_speaker_embedding and self.num_speakers > 0 else 0

    @property
    def interpolate_factor(self) -> float | None:  # vits.py:810
        if not self.encoder_sample_rate or not self.sample_rate:  # __post_init__ requires both or neither
            return None
        return self.sample_rate / self.encoder_sample_rate

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VitsModelConfig:
        """The inverse of to_dict. Unknown keys raise TypeError: this is our own file format."""
        d = dict(d)
        for key in ("resblock_kernel_sizes_decoder", "upsample_rates_decoder", "upsample_kernel_sizes_decoder"):
            if key in d:
                d[key] = _ints(d[key])
        if "resblock_dilation_sizes_decoder" in d:
            d["resblock_dilation_sizes_decoder"] = tuple(_ints(x) for x in d["resblock_dilation_sizes_decoder"])
        return cls(**d)

    @classmethod
    def coqui_keys(cls) -> frozenset[str]:
        """The Coqui `model_args` keys from_coqui reads, the ones it refuses included."""
        return frozenset({f.name for f in dataclasses.fields(cls)} - {"num_chars", "sample_rate"}) | frozenset(_REFUSED)

    @classmethod
    def from_coqui(
        cls, model_args: Mapping[str, Any], num_chars: int, *, sample_rate: int | None = None
    ) -> VitsModelConfig:
        """Read a Coqui `model_args` block. Keys the generator does not read are ignored."""
        for key, what in _REFUSED.items():
            if model_args.get(key):
                raise NotSupported(f"{what} are not supported ({key} is set)")
        known = {k: v for k, v in model_args.items() if k in cls.coqui_keys() and k not in _REFUSED}
        return cls.from_dict({**known, "num_chars": num_chars, "sample_rate": sample_rate})

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return {
            k: [list(x) for x in v]
            if k == "resblock_dilation_sizes_decoder"
            else list(v)
            if isinstance(v, tuple)
            else v
            for k, v in d.items()
        }
