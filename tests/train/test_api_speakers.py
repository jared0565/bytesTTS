# SPDX-License-Identifier: MPL-2.0
"""A multi-speaker voice package through the public API (needs torch to export the model)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import TEST_THREADS, TINY, TINY_ARGS, edit_json, repin, ser_db

from bytestts import BytesTTS
from bytestts.text.frontend import FrontendConfig, TextFrontend
from bytestts.train.vits.config import VitsModelConfig
from bytestts.train.vits.export import export_onnx
from bytestts.train.vits.model import VitsGenerator

SPEAKERS = {"alice": 0, "bob": 2, "carol": 1}


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> tuple[VitsGenerator, Path]:
    torch.manual_seed(11)
    args = {**TINY_ARGS, "use_sdp": False, "use_speaker_embedding": True, "num_speakers": 3}
    m = VitsGenerator(VitsModelConfig.from_coqui({**args, "speaker_embedding_channels": 8}, num_chars=39)).eval()
    d = tmp_path_factory.mktemp("multi") / "voice"
    shutil.copytree(TINY, d)
    export_onnx(m, d / "model.onnx")
    repin(d, "model.onnx")
    edit_json(d / "config.json", lambda c: c.update(inference_noise_scale=0.0, inference_noise_scale_dp=0.0))
    repin(d, "config.json")
    (d / "speakers.json").write_bytes(json.dumps(SPEAKERS).encode())
    repin(d, "speakers.json")
    edit_json(d / "voice.json", lambda v: v.update(speakers=["alice", "bob", "carol"], featured=["bob"]))
    return m, d


def torch_wav(m: VitsGenerator, text: str, sid: int) -> np.ndarray:
    cfg = FrontendConfig.from_voice_config(json.loads((TINY / "config.json").read_bytes()))
    (ids,) = TextFrontend(cfg).encode(text)
    wav, _ = m.infer(
        torch.tensor([ids]),
        torch.tensor([len(ids)]),
        noise_scale=0,
        length_scale=1,
        noise_scale_dp=0,
        sid=torch.tensor([sid]),
    )
    return wav[0, 0].numpy()


@pytest.mark.parametrize(("speaker", "sid"), [("alice", 0), ("bob", 2), ("carol", 1), (None, 2)])  # None: featured
def test_speaker_names_map_through_speakers_json(
    package: tuple[VitsGenerator, Path], speaker: str | None, sid: int
) -> None:
    m, d = package
    tts = BytesTTS.load_path(d, device="cpu", threads=TEST_THREADS)
    assert tts.info.speakers == ["alice", "bob", "carol"]
    wav = tts.synthesize("Hello there.", speaker=speaker).samples
    ref = torch_wav(m, "Hello there.", sid)
    assert len(wav) == len(ref) and ser_db(ref, wav) >= 50.0
