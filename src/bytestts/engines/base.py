# SPDX-License-Identifier: MPL-2.0
"""The engine seam: what every speech engine takes and returns."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

TokenBatch = Sequence[int]

MIN_SPEED, MAX_SPEED = 0.5, 2.0


@dataclass(frozen=True)
class VoiceRef:
    voice_id: str
    speaker: str | None = None
    reference_audio: Path | None = None  # voice cloning; no Phase 1 engine supports it


@dataclass(frozen=True)
class SynthesisOptions:
    speed: float = 1.0
    noise_scale: float | None = None  # None: the voice's default
    noise_scale_dp: float | None = None

    def __post_init__(self) -> None:
        if not MIN_SPEED <= self.speed <= MAX_SPEED:  # NaN fails this too
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, not {self.speed!r}")
        for name in ("noise_scale", "noise_scale_dp"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, not {value!r}")


class SpeechEngine(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def speakers(self) -> list[str]: ...

    def synthesize(self, tokens: TokenBatch, voice: VoiceRef, opts: SynthesisOptions) -> np.ndarray: ...
