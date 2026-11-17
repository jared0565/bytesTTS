# SPDX-License-Identifier: MPL-2.0
"""Phase 1 parity: the converted en-ljspeech voice against TTS-dev's golden trace (P0.7).

A heavy run (the development PC resets under sustained CPU load): deselected by default, selected by `-m parity`,
and skipped unless the local voice directories are given:

  BYTESTTS_PARITY_CONVERTED_DIR  convert_coqui.py's output (generator.safetensors, model_config.json)
  BYTESTTS_PARITY_VOICE_DIR      package_voice.py's output (model.onnx, config.json, voice.json)

The port runs at 1 torch thread (conftest) and the engine at threads=1. Bars are P0.9's: the port matches the
reference within max|d| <= 1e-3 and SER >= 60 dB; ONNX matches the port at SER >= 50 dB with equal durations.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from golden import GOLDEN_DIR, trace
from helpers import TEST_THREADS, ser_db

pytestmark = pytest.mark.parity
VOICE_ID = "en-ljspeech"
ZERO = {"noise_scale": 0.0, "length_scale": 1.0, "noise_scale_dp": 0.0}
SENTENCES: list[dict[str, Any]] = trace["sentences"]


def _dir(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is not set")
    return Path(value)


def _load_i16(name: str) -> np.ndarray:
    data: np.ndarray = np.load(GOLDEN_DIR / name, allow_pickle=False)
    assert data.dtype == np.int16, data.dtype
    return data.astype(np.float64) / 32768.0


@pytest.fixture(scope="module")
def port_out() -> list[tuple[np.ndarray, np.ndarray]]:
    """The port's zero-noise output for every trace sentence, computed once for both tests."""
    import torch

    from bytestts.train.vits.io import load_generator

    model = load_generator(_dir("BYTESTTS_PARITY_CONVERTED_DIR")).eval()
    out = []
    for s in SENTENCES:
        ids = s["token_ids"]
        wav, dur = model.infer(torch.tensor([ids]), torch.tensor([len(ids)]), **ZERO)
        out.append((wav[0, 0].numpy(), dur[0].numpy()))
    return out


@pytest.fixture(scope="module")
def engine() -> Any:
    import json

    from bytestts.engines.vits_onnx import VitsDefaults, VitsOnnxEngine

    pkg = _dir("BYTESTTS_PARITY_VOICE_DIR")
    config = json.loads((pkg / "config.json").read_bytes())
    return VitsOnnxEngine(pkg / "model.onnx", VitsDefaults.from_config(config), device="cpu", threads=TEST_THREADS)


def test_package_verifies() -> None:
    from bytestts.catalog.manifest import verify_package

    m = verify_package(_dir("BYTESTTS_PARITY_VOICE_DIR"))
    assert (m.id, m.language, m.engine, m.origin, m.sample_rate) == (VOICE_ID, "en", "vits-onnx", "catalog", 22050)


def test_trace_has_five_reference_waveforms() -> None:  # the bar below applies to these; a lost file must not pass
    assert [s["wav"] for s in SENTENCES if s.get("wav")] == [f"wav_{i:02d}.npy" for i in range(5)]


def test_port_matches_reference(port_out: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for i, (s, (wav, dur)) in enumerate(zip(SENTENCES, port_out, strict=True)):
        assert dur.tolist() == s["durations"], f"sentence {i}: durations differ"
        if s.get("wav"):
            ref = _load_i16(s["wav"])
            assert len(wav) == len(ref), f"sentence {i}: {len(wav)} samples, reference {len(ref)}"
            max_abs, ser = float(np.max(np.abs(ref - wav))), ser_db(ref, wav)
            assert max_abs <= 1e-3 and ser >= 60.0, f"sentence {i}: max|d| {max_abs:.2e}, SER {ser:.1f} dB"


def test_onnx_matches_port(port_out: list[tuple[np.ndarray, np.ndarray]], engine: Any) -> None:
    from bytestts.engines.base import SynthesisOptions, VoiceRef

    zero = SynthesisOptions(noise_scale=0.0, noise_scale_dp=0.0)
    for i, (s, (port_wav, port_dur)) in enumerate(zip(SENTENCES, port_out, strict=True)):
        wav, dur = engine.synthesize_with_durations(s["token_ids"], VoiceRef(VOICE_ID), zero)
        assert np.array_equal(dur, port_dur), f"sentence {i}: durations differ"
        ser = ser_db(port_wav, wav)
        assert ser >= 50.0, f"sentence {i}: SER {ser:.1f} dB"
