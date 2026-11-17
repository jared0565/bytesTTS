# SPDX-License-Identifier: MPL-2.0
"""The committed tiny VITS fixture (Task 10): consistent, pinned, small. Needs no torch."""

import hashlib
import json

import numpy as np
from helpers import CHAR_CFG, TINY, TINY_ARGS, ort_session, ser_db, tiny_golden

from bytestts.text.frontend import FrontendConfig

ZERO = np.array([0.0, 1.0, 0.0], np.float32)


def test_fixture_size() -> None:
    assert sum(f.stat().st_size for f in TINY.iterdir()) < 2_000_000


def test_voice_json_pins_every_listed_file() -> None:  # Task 12's verify_package relies on these bytes
    voice = json.loads((TINY / "voice.json").read_bytes())
    assert (
        voice["id"] == "tiny"
        and voice["engine"] == "vits-onnx"
        and set(voice["files"]) == {"model.onnx", "config.json"}
    )
    for name, entry in voice["files"].items():
        data = (TINY / name).read_bytes()
        assert (len(data), hashlib.sha256(data).hexdigest()) == (entry["size"], entry["sha256"]), name


def test_config_is_the_characters_frontend() -> None:
    cfg = json.loads((TINY / "config.json").read_bytes())
    assert FrontendConfig.from_voice_config(cfg) == CHAR_CFG
    assert cfg["sample_rate"] == json.loads((TINY / "voice.json").read_bytes())["sample_rate"]


def test_model_config_is_tiny_with_the_vocab_size() -> None:
    mc = json.loads((TINY / "model_config.json").read_bytes())
    assert mc["num_chars"] == len(CHAR_CFG.characters.vocab) == 39
    for key, value in TINY_ARGS.items():
        assert mc[key] == value, key
    assert mc["use_sdp"] is False  # the stochastic predictor alone is 5.3 MB: see tests/train/test_export.py


def test_fixture_onnx_reproduces_golden() -> None:
    g = tiny_golden()
    assert len(g["tokens"]) == 3
    sess = ort_session(TINY / "model.onnx")
    for tokens, dur, ref in zip(g["tokens"], g["durations"], g["wav"], strict=True):
        x = tokens[None, :].astype(np.int64)
        wav, d = sess.run(None, {"input": x, "input_lengths": np.array([x.shape[1]], np.int64), "scales": ZERO})
        assert np.array_equal(d[0], dur) and ser_db(ref, wav[0, 0]) >= 50.0
