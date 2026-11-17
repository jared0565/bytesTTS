# SPDX-License-Identifier: MPL-2.0
"""VITS on ONNX Runtime: the model exported by bytestts.train.vits.export (Task 10)."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort

from bytestts.engines.base import SynthesisOptions, TokenBatch, VoiceRef
from bytestts.errors import IntegrityError, NotSupported, VoiceNotFound

_log = logging.getLogger(__name__)

_CUDA, _CPU = "CUDAExecutionProvider", "CPUExecutionProvider"
_CUDA_ADVICE = (
    "the CUDA execution provider is listed but the session runs on CPU: install CUDA 13 and cuDNN 9, "
    "then reinstall onnxruntime-gpu after onnxruntime (P0.2: listed is not working)"
)

# --- external-data refusal --------------------------------------------------------------------------------
# A tensor stored as "external data" names a file that ONNX Runtime opens relative to the model, e.g.
# "../secret.bin". Core can't import `onnx`, so the protobuf wire format is read directly. Field numbers from
# onnx/onnx.proto at onnx 1.23.2 (the locked version):
#   ModelProto: graph = 7, training_info = 20, functions = 25
#   TrainingInfoProto: initialization = 1, algorithm = 2
#   FunctionProto: node = 7, attribute_proto = 11
#   GraphProto: node = 1, initializer = 5, sparse_initializer = 15
#   NodeProto: attribute = 5
#   AttributeProto: t = 5, g = 6, tensors = 10, graphs = 11, sparse_tensor = 22, sparse_tensors = 23
#   SparseTensorProto: values = 1, indices = 2
#   TensorProto: external_data = 13, data_location = 14 (DataLocation.EXTERNAL = 1)
# ONNX Runtime 1.32.0 has no session option that refuses external data (none of its 105 config keys does;
# session.model_external_initializers_file_folder_path only moves where the files are resolved), so this
# check is the guard. Behind it, the session is built from bytes with external data pinned to an empty private
# directory: a model loaded from bytes resolves external data against the CWD (measured: with the optimizer off,
# ORT read ./w.bin), and the pin makes any reference this scan missed resolve to nothing, at every level.
# The scan must read the model exactly as ORT will:
#   * ORT sniffs bytes 4..8 for "ORTM" and then loads its FlatBuffers format, which this scan cannot read.
#     Such bytes are refused here, and the session pins session.load_model_format=ONNX (measured: a real
#     .ort model loads without the pin and fails protobuf parsing with it).
#   * protobuf reads an enum varint as int32, so data_location 2**32 + 1 is EXTERNAL (measured in ORT).
#   * protobuf keeps a field key's low 32 bits: ORT loaded an initializer whose key carried 2**32 and read
#     its external file (measured), so keys are masked the same way before the field number is taken.
_CHILDREN: dict[str, dict[int, str]] = {
    "model": {7: "graph", 20: "training", 25: "function"},
    "training": {1: "graph", 2: "graph"},
    "function": {7: "node", 11: "attribute"},
    "graph": {1: "node", 5: "tensor", 15: "sparse"},
    "node": {5: "attribute"},
    "attribute": {5: "tensor", 6: "graph", 10: "tensor", 11: "graph", 22: "sparse", 23: "sparse"},
    "sparse": {1: "tensor", 2: "tensor"},
}
_EXTERNAL_DATA, _DATA_LOCATION, _LOCATION_EXTERNAL = 13, 14, 1
_ORT_FORMAT_ID = b"ORTM"  # the FlatBuffers file identifier, at bytes 4..8


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise IntegrityError("malformed ONNX model: truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift >= 64:
            raise IntegrityError("malformed ONNX model: varint longer than 64 bits")


def _fields(buf: memoryview) -> Iterator[tuple[int, int, int | memoryview]]:
    """(field number, wire type, value) for each field of one message; nested messages stay unparsed."""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        field, wire = (key & 0xFFFF_FFFF) >> 3, key & 7  # a key is uint32 to protobuf: high bits are dropped
        value: int | memoryview = 0
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            if length > end - pos:
                raise IntegrityError("malformed ONNX model: a field runs past the end of its message")
            value = buf[pos : pos + length]
            pos += length
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
            if pos > end:
                raise IntegrityError("malformed ONNX model: truncated fixed-width field")
        else:  # 3 and 4 are deprecated groups, 6 and 7 do not exist
            raise IntegrityError(f"malformed ONNX model: wire type {wire}")
        yield field, wire, value


def reject_external_data(model: bytes) -> None:
    """Raise IntegrityError unless the model is ONNX protobuf in which no tensor, at any depth, uses external data."""
    if bytes(model[4:8]) == _ORT_FORMAT_ID:
        raise IntegrityError("the model is in ORT format (FlatBuffers), which is refused: only ONNX models load")
    stack: list[tuple[str, memoryview]] = [("model", memoryview(model))]  # a stack, not recursion: depth is free
    while stack:
        kind, buf = stack.pop()
        for field, wire, value in _fields(buf):
            if kind == "tensor":
                location = value & 0xFFFF_FFFF if isinstance(value, int) else None  # as int32, as protobuf does
                if (field == _DATA_LOCATION and wire == 0 and location == _LOCATION_EXTERNAL) or (
                    field == _EXTERNAL_DATA and wire == 2
                ):
                    raise IntegrityError(
                        "the model stores tensors as external data (files outside the model), which is refused"
                    )
            elif wire == 2 and isinstance(value, memoryview):
                child = _CHILDREN[kind].get(field)
                if child is not None:
                    stack.append((child, value))


# --- the engine -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VitsDefaults:
    noise_scale: float
    length_scale: float
    noise_scale_dp: float
    sample_rate: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> VitsDefaults:
        """From a voice config.json; TTS-dev's defaults where it omits them (vits.py:569-572)."""

        def get(key: str, default: float) -> float:
            value = cfg.get(key)
            return default if value is None else float(value)

        return cls(
            get("inference_noise_scale", 0.667),
            get("length_scale", 1.0),
            get("inference_noise_scale_dp", 1.0),
            int(cfg["sample_rate"]),
        )


def _session_options(threads: int | None, external_data_dir: str) -> Any:
    options = ort.SessionOptions()
    options.add_session_config_entry("session.load_model_format", "ONNX")  # never sniff for FlatBuffers
    options.add_session_config_entry("session.model_external_initializers_file_folder_path", external_data_dir)
    if threads is not None:  # as spike/ort_safe.py
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


class VitsOnnxEngine:
    def __init__(
        self,
        model_path: Path,
        defaults: VitsDefaults,
        *,
        speakers: Mapping[str, int] | None = None,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        threads: int | None = None,
    ) -> None:
        if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
            raise ValueError(
                f"threads must be an int >= 1 or None, not {threads!r} (ONNX Runtime reads 0 as all cores)"
            )
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', not {device!r}")
        cuda_listed = _CUDA in ort.get_available_providers()
        if device == "cuda" and not cuda_listed:
            raise NotSupported("CUDA execution provider not available")
        providers = [_CUDA, _CPU] if device != "cpu" and cuda_listed else [_CPU]

        model = Path(model_path).read_bytes()
        reject_external_data(model)  # before ONNX Runtime sees a single byte
        with tempfile.TemporaryDirectory(prefix="bytestts-no-external-data-") as empty:  # read only while building
            self._session = ort.InferenceSession(
                model, sess_options=_session_options(threads, empty), providers=providers
            )

        actual = self.provider
        if device == "cuda" and actual != _CUDA:
            raise NotSupported(_CUDA_ADVICE)
        if device == "auto" and cuda_listed and actual != _CUDA:
            _log.warning("CUDA is listed but the session runs on %s; keeping CPU. %s", actual, _CUDA_ADVICE)

        self._needs_sid = "sid" in {i.name for i in self._session.get_inputs()}
        if self._needs_sid and not speakers:
            raise ValueError("this model takes a speaker id: pass its speakers mapping")
        if speakers and not self._needs_sid:
            raise ValueError("this model has a single speaker, but a speakers mapping was given")
        self._speakers = dict(speakers or {})
        self.defaults = defaults

    @property
    def provider(self) -> str:
        """The session's actual first execution provider, not the requested one."""
        return str(self._session.get_providers()[0])

    @property
    def sample_rate(self) -> int:
        return self.defaults.sample_rate

    @property
    def speakers(self) -> list[str]:
        return list(self._speakers)

    def _speaker_id(self, name: str | None) -> int:
        if not self._speakers:
            if name is not None:
                raise VoiceNotFound(f"speaker {name!r}: this voice has a single speaker")
            return 0
        if name is None:
            return next(iter(self._speakers.values()))  # the first listed speaker
        if name not in self._speakers:
            raise VoiceNotFound(f"speaker {name!r} is not in this voice")
        return self._speakers[name]

    def synthesize_with_durations(
        self, tokens: TokenBatch, voice: VoiceRef, opts: SynthesisOptions
    ) -> tuple[np.ndarray, np.ndarray]:
        """(waveform float32 [S], durations int64 [T]) for one token sequence."""
        if voice.reference_audio is not None:
            raise NotSupported("this engine cannot clone a voice from reference audio")
        sid = self._speaker_id(voice.speaker)
        ids = np.asarray(tokens)
        if ids.ndim != 1 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("tokens must be a non-empty sequence of integer ids")
        d = self.defaults
        scales = np.array(
            [
                d.noise_scale if opts.noise_scale is None else opts.noise_scale,
                d.length_scale / opts.speed,
                d.noise_scale_dp if opts.noise_scale_dp is None else opts.noise_scale_dp,
            ],
            np.float32,
        )
        feeds = {
            "input": ids.astype(np.int64)[None, :],
            "input_lengths": np.array([ids.size], np.int64),
            "scales": scales,
        }
        if self._needs_sid:
            feeds["sid"] = np.array([sid], np.int64)
        wav, durations = self._session.run(["output", "durations"], feeds)
        return np.ascontiguousarray(wav[0, 0], dtype=np.float32), durations[0]

    def synthesize(self, tokens: TokenBatch, voice: VoiceRef, opts: SynthesisOptions) -> np.ndarray:
        return self.synthesize_with_durations(tokens, voice, opts)[0]


ENGINES: dict[str, type] = {"vits-onnx": VitsOnnxEngine}
