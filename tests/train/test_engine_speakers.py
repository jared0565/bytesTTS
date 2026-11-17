# SPDX-License-Identifier: MPL-2.0
"""The ONNX engine's speaker mapping, on an exported tiny multi-speaker VITS (needs torch to export)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import TINY, TINY_ARGS, ser_db

from bytestts.engines.base import SynthesisOptions, VoiceRef
from bytestts.engines.vits_onnx import VitsDefaults, VitsOnnxEngine
from bytestts.errors import VoiceNotFound
from bytestts.train.vits.config import VitsModelConfig
from bytestts.train.vits.export import export_onnx
from bytestts.train.vits.model import VitsGenerator

DEFAULTS = VitsDefaults(0.0, 1.0, 0.0, 22050)
SPEAKERS = {"alice": 0, "bob": 2}
ZERO = SynthesisOptions(noise_scale=0, noise_scale_dp=0)
TOKENS = [5, 1, 9, 1, 14, 1, 3, 1, 22, 1, 7]


@pytest.fixture(scope="module")
def multi(tmp_path_factory: pytest.TempPathFactory) -> tuple[VitsGenerator, Path]:
    torch.manual_seed(7)
    cfg = {
        **TINY_ARGS,
        "use_sdp": False,
        "use_speaker_embedding": True,
        "num_speakers": 3,
        "speaker_embedding_channels": 8,
    }
    m = VitsGenerator(VitsModelConfig.from_coqui(cfg, num_chars=39)).eval()
    path = tmp_path_factory.mktemp("multi") / "m.onnx"
    export_onnx(m, path)
    return m, path


@pytest.mark.parametrize(("speaker", "sid"), [("bob", 2), ("alice", 0), (None, 0)])  # None: the first listed
def test_speaker_name_selects_its_id(multi: tuple[VitsGenerator, Path], speaker: str | None, sid: int) -> None:
    m, path = multi
    eng = VitsOnnxEngine(path, DEFAULTS, speakers=SPEAKERS, device="cpu", threads=1)
    wav, dur = eng.synthesize_with_durations(TOKENS, VoiceRef("multi", speaker=speaker), ZERO)
    tw, td = m.infer(
        torch.tensor([TOKENS]),
        torch.tensor([len(TOKENS)]),
        noise_scale=0,
        length_scale=1,
        noise_scale_dp=0,
        sid=torch.tensor([sid]),
    )
    assert np.array_equal(dur, td[0].numpy()) and ser_db(tw[0, 0].numpy(), wav) >= 50.0
    assert eng.speakers == ["alice", "bob"]


def test_unknown_speaker_raises(multi: tuple[VitsGenerator, Path]) -> None:
    eng = VitsOnnxEngine(multi[1], DEFAULTS, speakers=SPEAKERS, device="cpu", threads=1)
    with pytest.raises(VoiceNotFound):
        eng.synthesize(TOKENS, VoiceRef("multi", speaker="carol"), ZERO)


def test_multispeaker_model_needs_its_mapping(multi: tuple[VitsGenerator, Path]) -> None:
    with pytest.raises(ValueError, match="speakers mapping"):
        VitsOnnxEngine(multi[1], DEFAULTS, device="cpu", threads=1)


def test_single_speaker_model_refuses_a_mapping() -> None:
    defaults = VitsDefaults.from_config(json.loads((TINY / "config.json").read_bytes()))
    with pytest.raises(ValueError, match="single speaker"):
        VitsOnnxEngine(TINY / "model.onnx", defaults, speakers=SPEAKERS, device="cpu", threads=1)
