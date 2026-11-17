# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from helpers import HOSTILE, TINY_ARGS, ort_session, ser_db

from bytestts.train.vits.config import VitsModelConfig
from bytestts.train.vits.export import OPSET, export_onnx
from bytestts.train.vits.model import VitsGenerator

ZERO = np.array([0.0, 1.0, 0.0], np.float32)


def _model(**extra: Any) -> VitsGenerator:
    """Seeded, then perturbed so the zero-initialized flows are not identities (as the Task 9 differential)."""
    torch.manual_seed(1234)
    m = VitsGenerator(VitsModelConfig.from_coqui({**TINY_ARGS, **extra}, num_chars=39))
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return m.eval()


def _batch(lengths: list[int], seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, 39, (len(lengths), max(lengths)), generator=g)
    for i, n in enumerate(lengths):
        x[i, n:] = 0
    return x.numpy(), np.array(lengths, np.int64)


def _pre_ceil_margin(m: VitsGenerator, x: np.ndarray, lengths: np.ndarray, length_scale: float = 1.0) -> float:
    """Distance of the unrounded durations from the nearest integer: exact duration equality needs a margin."""
    with torch.no_grad():
        h, _, _, mask = m.text_encoder(torch.from_numpy(x), torch.from_numpy(lengths))
        if m.cfg.use_sdp:
            logw = m.duration_predictor(h, mask, reverse=True, noise_scale=0.0)
        else:
            logw = m.duration_predictor(h, mask)
        w = (torch.exp(logw) * mask * length_scale)[mask.bool()]
    return float(torch.min(torch.abs(w - torch.round(w))))


def _batch_with_margin(m: VitsGenerator, lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The first seeded batch whose unrounded durations sit >= 1e-3 from an integer (deterministic search)."""
    for seed in range(5, 105):
        x, lengths_np = _batch(lengths, seed)
        if _pre_ceil_margin(m, x, lengths_np) >= 1e-3:  # else ORT and torch may round a duration differently
            return x, lengths_np
    raise AssertionError("no batch with a duration margin in 100 seeds")


def _check_matches(m: VitsGenerator, path: Path, lengths: list[int], **feeds: np.ndarray) -> None:
    x, lengths_np = _batch_with_margin(m, lengths)
    wav, dur = ort_session(path).run(None, {"input": x, "input_lengths": lengths_np, "scales": ZERO, **feeds})
    sid = torch.from_numpy(feeds["sid"]) if "sid" in feeds else None
    tw, td = m.infer(
        torch.from_numpy(x), torch.from_numpy(lengths_np), noise_scale=0, length_scale=1, noise_scale_dp=0, sid=sid
    )
    assert dur.dtype == np.int64 and np.array_equal(dur, td.numpy())
    assert wav.shape == tuple(tw.shape) and ser_db(tw.numpy(), wav) >= 50.0


@pytest.mark.parametrize(
    "lengths", [[3], [5], [12], [31], [23, 9]], ids=["T3", "T5", "T12", "T31", "padded-batch"]
)  # T3/T5: shorter than the relative-attention window
def test_onnx_matches_torch_on_tiny(tmp_path: Path, lengths: list[int]) -> None:  # T differs from the dummy's
    m = _model(use_sdp=False)
    export_onnx(m, tmp_path / "m.onnx")
    _check_matches(m, tmp_path / "m.onnx", lengths)


def test_stochastic_predictor_export(tmp_path: Path) -> None:  # spline masking and RandomNormal, not committed
    m = _model()
    export_onnx(m, tmp_path / "sdp.onnx")
    _check_matches(m, tmp_path / "sdp.onnx", [19])
    _check_matches(m, tmp_path / "sdp.onnx", [27, 11])


def test_noise_scales_are_live_inputs(tmp_path: Path) -> None:  # TTS-dev's export baked them in (vits.py:1829)
    m = _model()
    export_onnx(m, tmp_path / "sdp.onnx")
    sess = ort_session(tmp_path / "sdp.onnx")
    x, lengths = _batch([33])  # not the dummy length: a shape baked into RandomNormal would fail here
    feeds = {"input": x, "input_lengths": lengths}
    quiet = sess.run(None, {**feeds, "scales": ZERO})[0]
    assert np.array_equal(quiet, sess.run(None, {**feeds, "scales": ZERO})[0])  # zero noise is deterministic
    noisy = sess.run(None, {**feeds, "scales": np.array([0.667, 1.0, 0.8], np.float32)})[0]
    assert noisy.shape != quiet.shape or not np.array_equal(noisy, quiet)


def test_length_scale_is_a_live_input(tmp_path: Path) -> None:
    m = _model(use_sdp=False)
    export_onnx(m, tmp_path / "m.onnx")
    x, lengths = _batch([15])
    sess = ort_session(tmp_path / "m.onnx")
    s1 = sess.run(None, {"input": x, "input_lengths": lengths, "scales": ZERO})[0].shape[-1]
    s2 = sess.run(None, {"input": x, "input_lengths": lengths, "scales": np.array([0, 2.0, 0], np.float32)})[0].shape[
        -1
    ]
    assert s2 > s1


def test_multispeaker_export_takes_sid(tmp_path: Path) -> None:
    m = _model(use_sdp=False, use_speaker_embedding=True, num_speakers=3, speaker_embedding_channels=8)
    export_onnx(m, tmp_path / "spk.onnx")
    assert [i.name for i in onnx.load(str(tmp_path / "spk.onnx")).graph.input][-1] == "sid"
    _check_matches(m, tmp_path / "spk.onnx", [14], sid=np.array([2], np.int64))


def test_io_names_and_opset(tmp_path: Path) -> None:
    export_onnx(_model(use_sdp=False), tmp_path / "m.onnx")
    model = onnx.load(str(tmp_path / "m.onnx"))
    assert OPSET == 17 and {o.domain or "ai.onnx": o.version for o in model.opset_import}["ai.onnx"] == OPSET
    assert [i.name for i in model.graph.input] == ["input", "input_lengths", "scales"]
    assert [o.name for o in model.graph.output] == ["output", "durations"]


def test_export_leaves_the_model_parametrized(tmp_path: Path) -> None:  # export works on a copy
    m = _model(use_sdp=False)
    keys = set(m.state_dict())
    export_onnx(m, tmp_path / "m.onnx")
    assert set(m.state_dict()) == keys and any(".parametrizations.weight.original0" in k for k in keys)


def test_hostile_external_data_fixture() -> None:  # Task 11 refuses this model before ONNX Runtime sees it
    model = onnx.load(str(HOSTILE / "external_data.onnx"), load_external_data=False)
    (init,) = model.graph.initializer
    assert init.data_location == onnx.TensorProto.EXTERNAL
    assert {e.key: e.value for e in init.external_data}["location"] == "../secret.bin"
