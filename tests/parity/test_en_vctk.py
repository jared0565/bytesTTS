# SPDX-License-Identifier: MPL-2.0
"""Parity for the VCTK English (UK) voice against TTS-dev's golden trace (2026-11-19, speaker p280).

A heavy run (the development PC resets under sustained CPU load): deselected by default, selected by `-m parity`,
and skipped unless the local voice directories are given:

  BYTESTTS_PARITY_ROOT  a directory holding en-vctk-converted/ (convert_coqui.py's output) and en-vctk/
                        (package_voice.py's output)

The port runs at 1 torch thread (conftest) and the engine at threads=1. Bars are P0.9's: the port matches the
reference within max|d| <= 1e-3 and SER >= 60 dB; ONNX matches the port at SER >= 50 dB with equal durations. The
voice has 109 speakers; the reference ran p280 (model id 54), so the port gets that id and the engine that NAME.
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
VOICE = "en-vctk"
ZERO = {"noise_scale": 0.0, "length_scale": 1.0, "noise_scale_dp": 0.0}
# The operator's shortlist (2026-11-19, by ear, UK accents): the first is the default speaker.
FEATURED = ["p228", "p232", "p229", "p243", "p244", "p287", "p262", "p247", "p304", "p253"]


def _dir(suffix: str = "") -> Path:
    root = os.environ.get("BYTESTTS_PARITY_ROOT")
    if not root:
        pytest.skip("BYTESTTS_PARITY_ROOT is not set")
    path = Path(root) / f"{VOICE}{suffix}"
    if not path.is_dir():
        pytest.skip(f"{path} does not exist")
    return path


@pytest.fixture(scope="module")
def trace() -> dict[str, Any]:
    return load_trace(VOICE)


def _load_i16(name: str) -> np.ndarray:
    data: np.ndarray = np.load(golden_dir(VOICE) / name, allow_pickle=False)
    assert data.dtype == np.int16, data.dtype
    return data.astype(np.float64) / 32768.0


@pytest.fixture(scope="module")
def port() -> Any:
    from bytestts.train.vits.io import load_generator

    return load_generator(_dir("-converted")).eval()


def _port_infer(model: Any, ids: list[int], sid: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    wav, dur = model.infer(torch.tensor([ids]), torch.tensor([len(ids)]), **ZERO, sid=torch.tensor([sid]))
    return wav[0, 0].numpy(), dur[0].numpy()


@pytest.fixture(scope="module")
def port_out(port: Any, trace: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """The port's zero-noise output for every trace sentence, with the reference's speaker id."""
    return [_port_infer(port, s["token_ids"], trace["speaker_id"]) for s in trace["sentences"]]


@pytest.fixture(scope="module")
def engine() -> Any:
    from bytestts.engines.vits_onnx import VitsDefaults, VitsOnnxEngine

    pkg = _dir()
    config = json.loads((pkg / "config.json").read_bytes())
    speakers = json.loads((pkg / "speakers.json").read_bytes())
    defaults = VitsDefaults.from_config(config)
    return VitsOnnxEngine(pkg / "model.onnx", defaults, device="cpu", threads=TEST_THREADS, speakers=speakers)


def _zero() -> Any:
    from bytestts.engines.base import SynthesisOptions

    return SynthesisOptions(noise_scale=0.0, noise_scale_dp=0.0)


def test_package_verifies() -> None:
    from bytestts.catalog.manifest import verify_package

    m = verify_package(_dir())
    assert (m.id, m.language, m.engine, m.origin, m.sample_rate) == (VOICE, "en-gb", "vits-onnx", "catalog", 22050)
    assert (m.licence, m.data_licence) == ("Apache-2.0", "CC-BY-4.0")


def test_package_lists_every_speaker_by_id(trace: dict[str, Any]) -> None:
    from bytestts.catalog.manifest import verify_package

    m = verify_package(_dir())
    assert len(m.speakers) == 109 and m.speakers[trace["speaker_id"]] == trace["speaker"]
    assert m.speakers[0] == "ED"  # the checkpoint's row 0 is "ED\n"; renamed when packaging (operator 2026-11-19)


def test_package_features_the_shortlist() -> None:  # a voice loaded by hand from its files defaults to featured[0]
    from bytestts.catalog.manifest import verify_package

    assert verify_package(_dir()).featured == FEATURED


def test_defaults_come_from_the_config() -> None:  # VCTK's scales, not TTS-dev's class defaults
    from bytestts.engines.vits_onnx import VitsDefaults

    config = json.loads((_dir() / "config.json").read_bytes())
    d = VitsDefaults.from_config(config)
    assert (d.noise_scale, d.length_scale, d.noise_scale_dp, d.sample_rate) == (0.667, 1.0, 0.8, 22050)


def test_trace_has_five_reference_waveforms(trace: dict[str, Any]) -> None:  # a lost file must not pass the bar
    assert [s["wav"] for s in trace["sentences"] if s.get("wav")] == [f"wav_{i:02d}.npy" for i in range(5)]


def test_port_matches_reference(trace: dict[str, Any], port_out: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for i, (s, (wav, dur)) in enumerate(zip(trace["sentences"], port_out, strict=True)):
        assert dur.tolist() == s["durations"], f"sentence {i}: durations differ"
        if s.get("wav"):
            ref = _load_i16(s["wav"])
            assert len(wav) == len(ref), f"sentence {i}: {len(wav)} samples, reference {len(ref)}"
            max_abs, ser = float(np.max(np.abs(ref - wav))), ser_db(ref, wav)
            assert max_abs <= 1e-3 and ser >= 60.0, f"sentence {i}: max|d| {max_abs:.2e}, SER {ser:.1f} dB"


def test_onnx_matches_port(trace: dict[str, Any], port_out: list[tuple[np.ndarray, np.ndarray]], engine: Any) -> None:
    from bytestts.engines.base import VoiceRef

    voice = VoiceRef(VOICE, speaker=trace["speaker"])  # by name, never the default speaker
    for i, (s, (port_wav, port_dur)) in enumerate(zip(trace["sentences"], port_out, strict=True)):
        wav, dur = engine.synthesize_with_durations(s["token_ids"], voice, _zero())
        assert np.array_equal(dur, port_dur), f"sentence {i}: durations differ"
        ser = ser_db(port_wav, wav)
        assert ser >= 50.0, f"sentence {i}: SER {ser:.1f} dB"


def test_the_name_selects_its_own_row(
    trace: dict[str, Any], port: Any, port_out: list[tuple[np.ndarray, np.ndarray]], engine: Any
) -> None:  # a name that reached the wrong row would still sound fluent; only this comparison shows it
    from bytestts.engines.base import VoiceRef

    ids = trace["sentences"][0]["token_ids"]
    wav, dur = engine.synthesize_with_durations(ids, VoiceRef(VOICE, speaker=trace["speaker"]), _zero())
    own_wav, own_dur = port_out[0]
    assert np.array_equal(dur, own_dur) and ser_db(own_wav, wav) >= 50.0
    other_wav, other_dur = _port_infer(port, ids, 0)
    n = min(len(other_wav), len(wav))
    assert not np.array_equal(dur, other_dur) or ser_db(other_wav[:n], wav[:n]) < 50.0
