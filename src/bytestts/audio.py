# SPDX-License-Identifier: MPL-2.0
"""Audio container and 16-bit PCM WAV writer (stdlib `wave` only)."""

from __future__ import annotations

import io
import os
import wave
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Audio:
    """Mono audio: 1-D float32 samples in [-1, 1] (values outside are clipped on write)."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray) or self.samples.ndim != 1 or self.samples.dtype != np.float32:
            raise ValueError("Audio.samples must be a 1-D float32 numpy array")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def duration(self) -> float:
        return float(self.samples.shape[0]) / self.sample_rate

    def to_wav_bytes(self) -> bytes:
        """16-bit PCM mono WAV, clipped to full scale. Same quantiser as the golden data writer."""
        pcm = np.clip(np.round(self.samples.astype(np.float64) * 32768), -32768, 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()

    def save(self, path: str | os.PathLike[str]) -> None:
        with open(path, "wb") as f:
            f.write(self.to_wav_bytes())


def concat(chunks: Sequence[Audio], pause_s: float = 0.0) -> Audio:
    """Join chunks with `pause_s` seconds of silence between them (never before or after)."""
    if not chunks:
        raise ValueError("concat needs at least one chunk")
    if pause_s < 0:
        raise ValueError(f"pause_s must be >= 0, got {pause_s}")
    rate = chunks[0].sample_rate
    if any(c.sample_rate != rate for c in chunks):
        raise ValueError("concat: chunks have different sample rates")
    gap = np.zeros(round(pause_s * rate), np.float32)
    parts: list[np.ndarray] = []
    for i, c in enumerate(chunks):
        if i:
            parts.append(gap)
        parts.append(c.samples)
    return Audio(np.concatenate(parts).astype(np.float32, copy=False), rate)
