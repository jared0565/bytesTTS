# SPDX-License-Identifier: MPL-2.0
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
from helpers import HOSTILE, TINY, ser_db, tiny_golden

from bytestts.engines.base import SynthesisOptions, VoiceRef
from bytestts.engines.vits_onnx import ENGINES, VitsDefaults, VitsOnnxEngine, reject_external_data
from bytestts.errors import IntegrityError, NotSupported, VoiceNotFound

MODEL = TINY / "model.onnx"
DEFAULTS = VitsDefaults.from_config(json.loads((TINY / "config.json").read_bytes()))
G = tiny_golden()


@pytest.fixture(scope="module")
def eng() -> VitsOnnxEngine:
    return VitsOnnxEngine(MODEL, DEFAULTS, device="cpu", threads=1)


def syn(eng: VitsOnnxEngine, speed: float) -> np.ndarray:
    return eng.synthesize(
        G["tokens"][1].tolist(), VoiceRef("tiny"), SynthesisOptions(speed=speed, noise_scale=0, noise_scale_dp=0)
    )


def test_matches_golden(eng: VitsOnnxEngine) -> None:
    for tokens, dur_ref, wav_ref in zip(G["tokens"], G["durations"], G["wav"], strict=True):
        wav, dur = eng.synthesize_with_durations(
            tokens.tolist(), VoiceRef("tiny"), SynthesisOptions(noise_scale=0, noise_scale_dp=0)
        )
        assert wav.dtype == np.float32 and wav.ndim == 1
        assert np.array_equal(dur, dur_ref) and ser_db(wav_ref, wav) >= 50.0


@pytest.mark.parametrize("speed", [0.49, 2.01, float("nan")])
def test_speed_bounds(speed: float) -> None:
    with pytest.raises(ValueError):
        SynthesisOptions(speed=speed)


def test_faster_is_shorter(eng: VitsOnnxEngine) -> None:
    assert len(syn(eng, 2.0)) < len(syn(eng, 1.0)) < len(syn(eng, 0.5))


def test_defaults_from_config() -> None:
    assert VitsDefaults.from_config({"sample_rate": 16000}) == VitsDefaults(0.667, 1.0, 1.0, 16000)  # vits.py:569-572
    assert DEFAULTS == VitsDefaults(0.667, 1.0, 1.0, 22050)


def test_engine_properties(eng: VitsOnnxEngine) -> None:
    assert (eng.sample_rate, eng.speakers, eng.provider) == (22050, [], "CPUExecutionProvider")
    assert ENGINES["vits-onnx"] is VitsOnnxEngine


def test_cuda_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(NotSupported):
        VitsOnnxEngine(MODEL, DEFAULTS, device="cuda")


@pytest.mark.filterwarnings("ignore:Specified provider:UserWarning")  # ORT notices the faked CUDA listing
def test_cuda_listed_but_not_working_raises(monkeypatch: pytest.MonkeyPatch) -> None:  # P0.2: listed is not working
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    with pytest.raises(NotSupported, match="CUDA 13"):
        VitsOnnxEngine(MODEL, DEFAULTS, device="cuda", threads=1)  # on a CPU-only install


@pytest.mark.filterwarnings("ignore:Specified provider:UserWarning")  # ORT notices the faked CUDA listing
def test_auto_falls_back_to_cpu_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    e = VitsOnnxEngine(MODEL, DEFAULTS, device="auto", threads=1)
    assert e.provider == "CPUExecutionProvider"
    assert sum(r.levelname == "WARNING" for r in caplog.records) == 1


def test_reference_audio_not_supported(eng: VitsOnnxEngine) -> None:
    with pytest.raises(NotSupported):
        eng.synthesize([1, 2], VoiceRef("tiny", reference_audio=Path("x.wav")), SynthesisOptions())


def test_unknown_speaker_raises(eng: VitsOnnxEngine) -> None:  # the tiny voice has one speaker
    with pytest.raises(VoiceNotFound):
        eng.synthesize([1, 2], VoiceRef("tiny", speaker="bob"), SynthesisOptions())


def test_empty_tokens_raise(eng: VitsOnnxEngine) -> None:
    with pytest.raises(ValueError):
        eng.synthesize([], VoiceRef("tiny"), SynthesisOptions())


def test_external_data_model_refused_before_ort(monkeypatch: pytest.MonkeyPatch) -> None:  # Task 10's hostile fixture
    calls: list[Any] = []
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: calls.append(a))
    with pytest.raises(IntegrityError):
        VitsOnnxEngine(HOSTILE / "external_data.onnx", DEFAULTS, device="cpu")
    assert calls == []  # ORT never saw the hostile model


def test_session_is_built_from_bytes_never_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def capture(model: Any, *a: Any, **k: Any) -> Any:
        seen.append(model)
        raise RuntimeError("stop")

    monkeypatch.setattr(ort, "InferenceSession", capture)
    with pytest.raises(RuntimeError, match="stop"):
        VitsOnnxEngine(MODEL, DEFAULTS, device="cpu", threads=1)
    assert len(seen) == 1 and isinstance(seen[0], bytes) and seen[0] == MODEL.read_bytes()


def test_tiny_model_passes_external_data_check() -> None:
    reject_external_data(MODEL.read_bytes())


def test_threads_applied() -> None:  # P0.9: the session's own options, not our argument
    o = VitsOnnxEngine(MODEL, DEFAULTS, device="cpu", threads=1)._session.get_session_options()
    assert o.intra_op_num_threads == 1 and o.inter_op_num_threads == 1
    assert o.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL


@pytest.mark.parametrize("n", [0, -1, True])
def test_threads_must_be_positive(n: Any) -> None:  # ORT reads 0 as "all cores"
    with pytest.raises(ValueError):
        VitsOnnxEngine(MODEL, DEFAULTS, device="cpu", threads=n)


@pytest.mark.gpu
def test_cuda_provider_used() -> None:  # deferred until hardware clearance (P0.9)
    assert VitsOnnxEngine(MODEL, DEFAULTS, device="cuda", threads=1).provider == "CUDAExecutionProvider"


# --- reject_external_data on hand-built wire bytes (no onnx package needed) ---


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:  # a length-delimited field
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


EXTERNAL_TENSOR = _varint(14 << 3) + _varint(1)  # TensorProto.data_location = EXTERNAL
EXTERNAL_ENTRY_ONLY = _ld(13, _ld(1, b"location") + _ld(2, b"../secret.bin"))  # TensorProto.external_data


@pytest.mark.parametrize(
    "model",
    [
        _ld(7, _ld(5, EXTERNAL_TENSOR)),  # graph.initializer
        _ld(7, _ld(5, EXTERNAL_ENTRY_ONLY)),  # external_data entries even without data_location
        _ld(7, _ld(15, _ld(1, EXTERNAL_TENSOR))),  # graph.sparse_initializer.values
        _ld(7, _ld(15, _ld(2, EXTERNAL_TENSOR))),  # graph.sparse_initializer.indices
        _ld(7, _ld(1, _ld(5, _ld(5, EXTERNAL_TENSOR)))),  # node.attribute.t
        _ld(7, _ld(1, _ld(5, _ld(10, EXTERNAL_TENSOR)))),  # node.attribute.tensors
        _ld(7, _ld(1, _ld(5, _ld(22, _ld(1, EXTERNAL_TENSOR))))),  # node.attribute.sparse_tensor
        _ld(7, _ld(1, _ld(5, _ld(6, _ld(5, EXTERNAL_TENSOR))))),  # node.attribute.g (If/Loop body) initializer
        _ld(7, _ld(1, _ld(5, _ld(11, _ld(5, EXTERNAL_TENSOR))))),  # node.attribute.graphs
        _ld(20, _ld(1, _ld(5, EXTERNAL_TENSOR))),  # training_info.initialization
        _ld(20, _ld(2, _ld(5, EXTERNAL_TENSOR))),  # training_info.algorithm
        _ld(25, _ld(7, _ld(5, _ld(5, EXTERNAL_TENSOR)))),  # functions.node.attribute.t
        _ld(25, _ld(11, _ld(5, EXTERNAL_TENSOR))),  # functions.attribute_proto.t (a default)
    ],
    ids=lambda b: f"{len(b)}B",
)
def test_external_data_found_wherever_it_hides(model: bytes) -> None:
    with pytest.raises(IntegrityError, match="external data"):
        reject_external_data(model)


def test_deep_nesting_is_walked_without_recursion() -> None:  # a hostile model must not hit the recursion limit
    graph = _ld(5, EXTERNAL_TENSOR)
    for _ in range(5000):
        graph = _ld(1, _ld(5, _ld(6, graph)))  # node.attribute.g -> graph, 5000 levels
    with pytest.raises(IntegrityError, match="external data"):
        reject_external_data(_ld(7, graph))


@pytest.mark.parametrize("bad", [b"\x3a\xff", b"\x3a\x05ab", b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff", b"\x3b"])
def test_malformed_model_raises_integrity_error(bad: bytes) -> None:  # truncated, overlong, bad varint, group
    with pytest.raises(IntegrityError):
        reject_external_data(bad)


def test_raw_data_is_skipped_not_parsed() -> None:  # tensor bytes that look like fields must not confuse the walk
    tensor = _ld(9, EXTERNAL_TENSOR * 3)  # TensorProto.raw_data whose content mimics data_location
    reject_external_data(_ld(7, _ld(5, tensor)))
