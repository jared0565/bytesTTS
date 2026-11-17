# SPDX-License-Identifier: MPL-2.0
"""Shared test helpers. Imported as `helpers` (pyproject pythonpath = tests), never from conftest.py:
under --import-mode=importlib, importing from conftest.py creates a second copy of the module."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bytestts.errors import FrontendUnavailable
from bytestts.text.frontend import FrontendConfig
from bytestts.text.phonemizers.espeak import EspeakInstall, EspeakPhonemizer, find_espeak
from bytestts.text.tokenizer import CharacterSet

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = FIXTURES / "tiny_vits"
HOSTILE = FIXTURES / "hostile"

# The characters frontend of the tiny fixture (Task 8 defined it; Task 10's fixture config.json must equal it).
CHAR_CFG = FrontendConfig(
    kind="characters",
    language="en",
    phoneme_language=None,
    cleaner="basic_cleaners",
    characters=CharacterSet(
        characters="abcdefghijklmnopqrstuvwxyz",
        punctuations="!'(),-.:;? ",
        pad="<PAD>",
        eos=None,
        bos=None,
        blank="<BLNK>",
        is_unique=True,
        is_sorted=True,
    ),
    add_blank=True,
    use_eos_bos=False,
)

# A tiny VITS in Coqui `model_args` form: hidden 16, 1 layer, upsample [4, 4] (Task 9).
TINY_ARGS: dict[str, Any] = {
    "out_channels": 17,
    "hidden_channels": 16,
    "hidden_channels_ffn_text_encoder": 32,
    "num_heads_text_encoder": 2,
    "num_layers_text_encoder": 1,
    "num_layers_posterior_encoder": 1,
    "num_layers_flow": 1,
    "resblock_kernel_sizes_decoder": [3],
    "resblock_dilation_sizes_decoder": [[1, 3, 5]],
    "upsample_rates_decoder": [4, 4],
    "upsample_initial_channel_decoder": 32,
    "upsample_kernel_sizes_decoder": [8, 8],
}


def ser_db(ref: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-error ratio of `test` against `ref`, in dB."""
    ref64 = np.asarray(ref, dtype=np.float64)
    err = np.sum((ref64 - np.asarray(test, dtype=np.float64)) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(np.sum(ref64**2) / err))


def ort_session(model: str | Path | bytes) -> Any:
    """An ONNX Runtime CPU session at 1 thread, spinning off (spike/ort_safe.py): what conftest.py demands."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    src = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(src, sess_options=so, providers=["CPUExecutionProvider"])


def tiny_golden() -> dict[str, list[np.ndarray]]:
    """golden.npz as lists, indexable like the plan's G["tokens"][0]. Stored per index: no pickle."""
    with np.load(TINY / "golden.npz", allow_pickle=False) as z:
        n = int(z["count"])
        return {key: [z[f"{key}_{i}"] for i in range(n)] for key in ("tokens", "durations", "wav")}


def require_espeak(expected_version: str | None) -> EspeakInstall:
    """The espeak-ng install the tests should use. Skips when it is missing or not `expected_version`,
    but FAILS instead when BYTESTTS_REQUIRE_ESPEAK=1 (CI, the matrix), so a gap can't pass unseen."""
    required = os.environ.get("BYTESTTS_REQUIRE_ESPEAK") == "1"
    try:
        install = find_espeak()
        version = EspeakPhonemizer("en-us", install=install).version
    except FrontendUnavailable as e:
        reason = f"espeak-ng unavailable: {e}"
        if required:
            pytest.fail(reason)
        pytest.skip(reason)
    if expected_version is not None and version != expected_version:
        reason = f"espeak-ng {version} found, tests need the pinned {expected_version}"
        if required:
            pytest.fail(reason)
        pytest.skip(reason)
    return install
