# SPDX-License-Identifier: MPL-2.0
import math
from pathlib import Path

import pytest
import torch
from helpers import TINY_ARGS
from safetensors.torch import load_file, save_file

from bytestts.errors import NotSupported
from bytestts.train.vits.config import VitsModelConfig
from bytestts.train.vits.io import load_generator, save_generator
from bytestts.train.vits.model import VitsGenerator

TINY = VitsModelConfig.from_coqui(TINY_ARGS, num_chars=40)
HOP = 16  # 4 * 4
ZERO = {"noise_scale": 0.0, "length_scale": 1.0, "noise_scale_dp": 0.0}


def _model(cfg: VitsModelConfig = TINY, seed: int = 0) -> VitsGenerator:
    torch.manual_seed(seed)
    return VitsGenerator(cfg).eval()


def _tokens(n: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return torch.randint(1, 40, (1, n), generator=g), torch.tensor([n])


def test_torch_runs_single_threaded() -> None:  # tests/train/conftest.py
    assert torch.get_num_threads() == 1


def test_roundtrip_strict(tmp_path: Path) -> None:
    m = _model()
    save_generator(m, tmp_path)
    m2 = load_generator(tmp_path)
    assert m.state_dict().keys() == m2.state_dict().keys()
    for k, v in m.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k
    assert m2.cfg == TINY and not m2.training


def test_load_rejects_unexpected_keys(tmp_path: Path) -> None:
    save_generator(_model(), tmp_path)
    sd = load_file(tmp_path / "generator.safetensors")
    sd["disc.extra.weight"] = torch.zeros(1)
    save_file(sd, tmp_path / "generator.safetensors")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_generator(tmp_path)


def test_infer_deterministic_at_zero_noise() -> None:
    m, (x, lengths) = _model(), _tokens()
    a, b = m.infer(x, lengths, **ZERO), m.infer(x, lengths, **ZERO)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[1].dtype == torch.int64 and a[1].shape == (1, 12)
    assert a[0].ndim == 3 and a[0].shape[:2] == (1, 1)


def test_durations_account_for_every_sample() -> None:
    wav, dur = _model().infer(*_tokens(), **ZERO)
    assert wav.shape[-1] == max(int(dur.sum()), 1) * HOP


def test_length_scale_lengthens() -> None:
    m, (x, lengths) = _model(), _tokens()
    s1 = m.infer(x, lengths, **ZERO)[0].shape[-1]
    s2 = m.infer(x, lengths, **{**ZERO, "length_scale": 2.0})[0].shape[-1]
    assert s2 > s1


def test_max_inference_len_crops_the_decoder_input() -> None:
    m = _model(VitsModelConfig.from_coqui({**TINY_ARGS, "max_inference_len": 5}, num_chars=40))
    wav, dur = m.infer(*_tokens(), **ZERO)
    assert int(dur.sum()) > 5 and wav.shape[-1] == 5 * HOP


def test_encoder_sample_rate_upsamples_z() -> None:  # vits.py:944 upsampling_z, factor 16000/8000 = 2
    plain = _model()
    up = VitsGenerator(VitsModelConfig.from_coqui({**TINY_ARGS, "encoder_sample_rate": 8000}, 40, sample_rate=16000))
    up.load_state_dict(plain.state_dict(), strict=True)  # same keys: the resampler is training-only
    up.eval()
    x, lengths = _tokens()
    assert up.infer(x, lengths, **ZERO)[0].shape[-1] == 2 * plain.infer(x, lengths, **ZERO)[0].shape[-1]


def test_encoder_sample_rate_needs_the_audio_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        VitsModelConfig.from_coqui({**TINY_ARGS, "encoder_sample_rate": 8000}, num_chars=40)


def test_coqui_key_names_present() -> None:
    keys = VitsGenerator(TINY).state_dict().keys()
    for prefix in ("text_encoder.", "posterior_encoder.", "flow.", "duration_predictor.", "waveform_decoder."):
        assert any(k.startswith(prefix) for k in keys), prefix
    assert not any(k.startswith("emb_g.") for k in keys)
    assert any(".parametrizations.weight.original0" in k for k in keys)  # weight norm, parametrized
    assert "waveform_decoder.conv_pre.weight" in keys and "waveform_decoder.conv_post.weight" in keys  # vits.py:715


def test_multispeaker_has_emb_g_and_needs_sid() -> None:
    cfg = VitsModelConfig.from_coqui(
        {**TINY_ARGS, "use_speaker_embedding": True, "num_speakers": 3, "speaker_embedding_channels": 8}, num_chars=40
    )
    m = _model(cfg)
    assert m.state_dict()["emb_g.weight"].shape == (3, 8)
    x, lengths = _tokens()
    with pytest.raises(ValueError, match="speaker"):
        m.infer(x, lengths, **ZERO)
    a = m.infer(x, lengths, **ZERO, sid=torch.tensor([0]))[0]
    b = m.infer(x, lengths, **ZERO, sid=torch.tensor([2]))[0]
    assert not torch.equal(a, b)


@pytest.mark.parametrize("key", ["use_d_vector_file", "use_language_embedding"])
def test_unported_conditioning_is_refused(key: str) -> None:
    with pytest.raises(NotSupported):
        VitsModelConfig.from_coqui({**TINY_ARGS, key: True}, num_chars=40)


def test_defaults_are_vitsargs_defaults() -> None:  # TTS-dev vits.py:366-600
    d = VitsModelConfig.from_coqui({}, num_chars=10)
    assert (d.hidden_channels, d.num_layers_text_encoder, d.out_channels, d.use_sdp) == (192, 6, 513, True)
    assert d.upsample_rates_decoder == (8, 8, 2, 2) and d.resblock_dilation_sizes_decoder == ((1, 3, 5),) * 3
    assert (d.inference_noise_scale, d.inference_noise_scale_dp, d.length_scale) == (0.667, 1.0, 1)
    assert d.max_inference_len is None and math.isclose(d.dropout_p_duration_predictor, 0.5)


def test_config_dict_roundtrip() -> None:
    assert VitsModelConfig.from_dict(TINY.to_dict()) == TINY


def test_unknown_coqui_keys_are_ignored() -> None:  # Coqui model_args carry many training-only keys
    assert VitsModelConfig.from_coqui({**TINY_ARGS, "some_training_only_key": True}, num_chars=40) == TINY
