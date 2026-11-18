# SPDX-License-Identifier: MPL-2.0
"""Parity for the CSS10 Spanish and French voices against TTS-dev's golden traces (2026-11-18).

A heavy run (the development PC resets under sustained CPU load): deselected by default, selected by `-m parity`,
and skipped unless the local voice directories are given:

  BYTESTTS_PARITY_ROOT  a directory holding, per voice, <id>-converted/ (convert_coqui.py's output) and <id>/
                        (package_voice.py's output)

The port runs at 1 torch thread (conftest) and the engine at threads=1. Bars are P0.9's, as for en-ljspeech: the
port matches the reference within max|d| <= 1e-3 and SER >= 60 dB; ONNX matches the port at SER >= 50 dB with equal
durations. Both voices have a one-row speaker table: the port gets speaker id 0, and the engine its default speaker.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from golden import golden_dir, load_trace
from helpers import TEST_THREADS, ser_db

pytestmark = pytest.mark.parity
VOICES = {"es-css10": ("es", "tux"), "fr-css10": ("fr", "gilles")}
ZERO = {"noise_scale": 0.0, "length_scale": 1.0, "noise_scale_dp": 0.0}


def _dir(voice_id: str, suffix: str = "") -> Path:
    root = os.environ.get("BYTESTTS_PARITY_ROOT")
    if not root:
        pytest.skip("BYTESTTS_PARITY_ROOT is not set")
    path = Path(root) / f"{voice_id}{suffix}"
    if not path.is_dir():
        pytest.skip(f"{path} does not exist")
    return path


@pytest.fixture(scope="module", params=sorted(VOICES))
def voice_id(request: pytest.FixtureRequest) -> str:
    value: str = request.param
    return value


@pytest.fixture(scope="module")
def trace(voice_id: str) -> dict[str, Any]:
    return load_trace(voice_id)


def _load_i16(voice_id: str, name: str) -> np.ndarray:
    data: np.ndarray = np.load(golden_dir(voice_id) / name, allow_pickle=False)
    assert data.dtype == np.int16, data.dtype
    return data.astype(np.float64) / 32768.0


@pytest.fixture(scope="module")
def port_out(voice_id: str, trace: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """The port's zero-noise output for every trace sentence, with the reference's speaker id."""
    import torch

    from bytestts.train.vits.io import load_generator

    model = load_generator(_dir(voice_id, "-converted")).eval()
    sid = torch.tensor([trace["speaker_id"]])
    out = []
    for s in trace["sentences"]:
        ids = s["token_ids"]
        wav, dur = model.infer(torch.tensor([ids]), torch.tensor([len(ids)]), **ZERO, sid=sid)
        out.append((wav[0, 0].numpy(), dur[0].numpy()))
    return out


@pytest.fixture(scope="module")
def engine(voice_id: str) -> Any:
    from bytestts.engines.vits_onnx import VitsDefaults, VitsOnnxEngine

    pkg = _dir(voice_id)
    config = json.loads((pkg / "config.json").read_bytes())
    speakers = json.loads((pkg / "speakers.json").read_bytes())
    defaults = VitsDefaults.from_config(config)
    return VitsOnnxEngine(pkg / "model.onnx", defaults, device="cpu", threads=TEST_THREADS, speakers=speakers)


def test_package_verifies(voice_id: str) -> None:
    from bytestts.catalog.manifest import verify_package

    language, speaker = VOICES[voice_id]
    m = verify_package(_dir(voice_id))
    assert (m.id, m.language, m.engine, m.origin, m.sample_rate) == (voice_id, language, "vits-onnx", "catalog", 22050)
    assert m.speakers == [speaker] and (m.licence, m.data_licence) == ("BSD-3-Clause", "CC0-1.0")


def test_defaults_come_from_the_config(voice_id: str) -> None:  # CSS10's scales, not TTS-dev's class defaults
    from bytestts.engines.vits_onnx import VitsDefaults

    config = json.loads((_dir(voice_id) / "config.json").read_bytes())
    d = VitsDefaults.from_config(config)
    assert (d.noise_scale, d.length_scale, d.noise_scale_dp, d.sample_rate) == (0.3, 1.0, 0.3, 22050)


def test_trace_has_five_reference_waveforms(trace: dict[str, Any]) -> None:  # a lost file must not pass the bar
    assert [s["wav"] for s in trace["sentences"] if s.get("wav")] == [f"wav_{i:02d}.npy" for i in range(5)]


def test_port_matches_reference(
    voice_id: str, trace: dict[str, Any], port_out: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    for i, (s, (wav, dur)) in enumerate(zip(trace["sentences"], port_out, strict=True)):
        assert dur.tolist() == s["durations"], f"sentence {i}: durations differ"
        if s.get("wav"):
            ref = _load_i16(voice_id, s["wav"])
            assert len(wav) == len(ref), f"sentence {i}: {len(wav)} samples, reference {len(ref)}"
            max_abs, ser = float(np.max(np.abs(ref - wav))), ser_db(ref, wav)
            assert max_abs <= 1e-3 and ser >= 60.0, f"sentence {i}: max|d| {max_abs:.2e}, SER {ser:.1f} dB"


def test_onnx_matches_port(
    voice_id: str, trace: dict[str, Any], port_out: list[tuple[np.ndarray, np.ndarray]], engine: Any
) -> None:
    from bytestts.engines.base import SynthesisOptions, VoiceRef

    zero = SynthesisOptions(noise_scale=0.0, noise_scale_dp=0.0)
    for i, (s, (port_wav, port_dur)) in enumerate(zip(trace["sentences"], port_out, strict=True)):
        wav, dur = engine.synthesize_with_durations(s["token_ids"], VoiceRef(voice_id), zero)  # the default speaker
        assert np.array_equal(dur, port_dur), f"sentence {i}: durations differ"
        ser = ser_db(port_wav, wav)
        assert ser >= 50.0, f"sentence {i}: SER {ser:.1f} dB"
