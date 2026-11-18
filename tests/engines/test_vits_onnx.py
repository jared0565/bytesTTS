# SPDX-License-Identifier: MPL-2.0
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
from helpers import HOSTILE, TINY, ser_db, tiny_golden

from bytestts.engines.base import SynthesisOptions, VoiceRef
from bytestts.engines.vits_onnx import ENGINES, VitsDefaults, VitsOnnxEngine, _session_options, reject_external_data
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
    with pytest.raises(NotSupported, match="CUDA 13") as caught:
        VitsOnnxEngine(MODEL, DEFAULTS, device="cuda", threads=1)  # on a CPU-only install
    assert "P0." not in str(caught.value)  # users get advice, not the project's internal plan references


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


# --- parser differentials: what the scan reads must be what ONNX Runtime loads ---


@pytest.fixture(scope="module")
def ort_format_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny model in ORT's FlatBuffers format, written by ONNX Runtime itself (one single-threaded session)."""
    path = tmp_path_factory.mktemp("ortfmt") / "tiny.ort"
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.optimized_model_filepath = str(path)
    so.add_session_config_entry("session.save_model_format", "ORT")
    ort.InferenceSession(str(MODEL), sess_options=so, providers=["CPUExecutionProvider"])
    return path


def test_ort_format_model_refused_before_ort(ort_format_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ORT loads FlatBuffers models (identifier ORTM) by sniffing, and the protobuf scan cannot read them
    assert ort_format_model.read_bytes()[4:8] == b"ORTM"
    calls: list[Any] = []
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: calls.append(a))
    with pytest.raises(IntegrityError, match="ORT format"):
        VitsOnnxEngine(ort_format_model, DEFAULTS, device="cpu", threads=1)
    assert calls == []


def test_session_parses_only_onnx_protobuf(ort_format_model: Path) -> None:
    # the second layer, independent of the identifier check: the session never takes the FlatBuffers path
    with pytest.raises(Exception, match="protobuf parsing failed"):
        ort.InferenceSession(
            ort_format_model.read_bytes(),
            sess_options=_session_options(1, str(ort_format_model.parent)),
            providers=["CPUExecutionProvider"],
        )
    for threads in (1, None):
        assert _session_options(threads, "x").get_session_config_entry("session.load_model_format") == "ONNX"


@pytest.mark.parametrize("value", [2**32 + 1, 2**63 + 1, 2**64 + 1])
def test_data_location_is_read_as_int32(value: int) -> None:
    # protobuf truncates an enum varint to 32 bits: on raw bytes, ONNX Runtime 1.32.0 read the external file for
    # data_location 2**32 + 1 and refused it for 2**32 (measured, with both controls firing)
    with pytest.raises(IntegrityError, match="external data"):
        reject_external_data(_ld(7, _ld(5, _varint(14 << 3) + _varint(value))))


@pytest.mark.parametrize("high", [1 << 32, 1 << 40, 1 << 63])
def test_field_keys_are_read_as_uint32(high: int) -> None:
    # protobuf keeps a key's low 32 bits: ONNX Runtime 1.32.0 loaded an initializer whose key carried 2**32 and
    # read its external file (measured). Every key here carries `high`, the external_data entry included.
    entry = _varint(high | 13 << 3 | 2) + _varint(10) + _ld(1, b"location")
    tensor = entry + _varint(high | 14 << 3) + _varint(1)
    init = _varint(high | 5 << 3 | 2) + _varint(len(tensor)) + tensor
    model = _varint(high | 7 << 3 | 2) + _varint(len(init)) + init
    with pytest.raises(IntegrityError, match="external data"):
        reject_external_data(model)


def _vi(field: int, value: int) -> bytes:  # a varint field
    return _varint(field << 3) + _varint(value)


def _external_add_model(location: bytes) -> bytes:
    """y = x + w, with w one float32 stored as external data at `location`; hand-built, so no onnx needed."""
    float1 = _ld(1, _vi(1, 1) + _ld(2, _ld(1, _vi(1, 1))))  # TypeProto.tensor_type: FLOAT, shape [1]
    w = _vi(1, 1) + _vi(2, 1) + _ld(8, b"w")  # dims [1], FLOAT, name
    for key, value in ((b"location", location), (b"offset", b"0"), (b"length", b"4")):
        w += _ld(13, _ld(1, key) + _ld(2, value))
    w += _vi(14, 1)  # data_location EXTERNAL
    node = _ld(1, b"x") + _ld(1, b"w") + _ld(2, b"y") + _ld(4, b"Add")
    graph = (
        _ld(1, node)
        + _ld(2, b"g")
        + _ld(5, w)
        + _ld(11, _ld(1, b"x") + _ld(2, float1))
        + _ld(12, _ld(1, b"y") + _ld(2, float1))
    )
    return _vi(1, 8) + _ld(8, _ld(1, b"") + _vi(2, 17)) + _ld(7, graph)  # ir_version 8, opset 17


def test_bytes_session_never_reads_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A model loaded from bytes resolves external data against the cwd: ONNX Runtime 1.32.0 read ./w.bin with the
    # optimizer off (measured). The folder pin must refuse it whatever the optimization level.
    (tmp_path / "w.bin").write_bytes(struct.pack("<f", 7.0))
    monkeypatch.chdir(tmp_path)
    model = _external_add_model(b"w.bin")
    with pytest.raises(IntegrityError):
        reject_external_data(model)  # the first layer
    empty = tmp_path / "empty"
    empty.mkdir()
    off = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    x = {"x": np.zeros(1, np.float32)}

    control = ort.SessionOptions()  # without the pin: proves the model is a working external-data model
    control.intra_op_num_threads = 1
    control.graph_optimization_level = off
    assert ort.InferenceSession(model, sess_options=control, providers=["CPUExecutionProvider"]).run(None, x)[0] == 7.0

    for level in (off, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
        options = _session_options(1, str(empty))
        options.graph_optimization_level = level
        with pytest.raises(Exception, match="does not exist"):
            ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])


def test_engine_pins_external_data_to_an_empty_private_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Path, list[Path]]] = []

    def capture(model: Any, sess_options: Any, **k: Any) -> Any:
        folder = Path(sess_options.get_session_config_entry("session.model_external_initializers_file_folder_path"))
        seen.append((folder, list(folder.iterdir())))
        raise RuntimeError("stop")

    monkeypatch.setattr(ort, "InferenceSession", capture)
    with pytest.raises(RuntimeError, match="stop"):
        VitsOnnxEngine(MODEL, DEFAULTS, device="cpu", threads=1)
    ((folder, contents),) = seen
    assert contents == [] and folder != Path.cwd() and folder != MODEL.parent
    assert not folder.exists()  # removed once the session is built
