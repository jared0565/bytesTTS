# SPDX-License-Identifier: MPL-2.0
import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from helpers import FIXTURES

from bytestts.errors import NotSupported
from bytestts.train.vits.config import VitsModelConfig

ROOT = Path(__file__).resolve().parents[2]
COQUI = FIXTURES / "coqui"


def load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "convert" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ck = load_tool("coqui_keys")


def ljspeech() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((COQUI / "en-ljspeech.config.json").read_bytes())
    return data


# --- state dicts ---


def test_split() -> None:
    g, d = ck.split_state_dict({"text_encoder.w": 1, "disc.x": 2, "speaker_encoder.y": 3})
    assert g == {"text_encoder.w": 1} and d == {"x": 2}


def test_split_drops_the_resampler_kernel() -> None:  # torchaudio's buffer, not part of the port (Task 9 finding)
    g, _ = ck.split_state_dict({"audio_resampler.kernel": 1, "flow.w": 2})
    assert g == {"flow.w": 2}


def test_split_matches_prefixes_not_substrings() -> None:
    g, d = ck.split_state_dict({"text_encoder.disc.x": 1, "discriminator.y": 2})
    assert g == {"text_encoder.disc.x": 1, "discriminator.y": 2} and d == {}


def test_rename() -> None:
    renamed = ck.rename_legacy_weight_norm({"a.weight_g": 1, "a.weight_v": 2, "b.bias": 3})
    assert renamed == {"a.parametrizations.weight.original0": 1, "a.parametrizations.weight.original1": 2, "b.bias": 3}


@pytest.mark.parametrize(
    "sd",
    [
        {"a.weight_g": 1, "a.weight_v": 2, "a.parametrizations.weight.original0": 3},  # two keys for one name
        {"a.weight_g": 1},  # half a weight norm
        {"a.weight_v": 2},
    ],
)
def test_rename_refuses_ambiguous_input(sd: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ck.rename_legacy_weight_norm(sd)


# --- voice config ---


def test_extract_voice_config_reproduces_phase0() -> None:  # the P0.7 trace, from the same config
    vc = ck.extract_voice_config(ljspeech(), "en")
    vc.pop("model_args")
    assert vc == json.loads((COQUI / "en-ljspeech.voice_config.json").read_bytes())


def test_model_args_are_what_the_generator_reads() -> None:
    model_args = ck.extract_voice_config(ljspeech(), "en")["model_args"]
    assert "hidden_channels" in model_args and "use_sdp" in model_args
    assert not {"init_discriminator", "num_chars", "use_spectral_norm_disriminator"} & set(model_args)  # not read
    assert set(model_args) == set(ljspeech()["model_args"]) & VitsModelConfig.coqui_keys()
    cfg = VitsModelConfig.from_coqui(model_args, num_chars=179, sample_rate=22050)
    assert cfg.hidden_channels == ljspeech()["model_args"]["hidden_channels"]


@pytest.mark.parametrize("key", ["use_d_vector_file", "use_language_embedding"])
def test_refusals_survive_extraction(key: str) -> None:  # else from_coqui could not refuse them later
    cfg = ljspeech()
    cfg["model_args"][key] = True
    with pytest.raises(NotSupported):
        VitsModelConfig.from_coqui(ck.extract_voice_config(cfg, "en")["model_args"], num_chars=179)


def test_top_level_scales_win_over_model_args() -> None:  # the P0.7 rule; v0.6.1 configs have only model_args
    cfg = ljspeech()
    assert ck.extract_voice_config(cfg, "en")["inference_noise_scale_dp"] == 0.8
    cfg = copy.deepcopy(cfg)
    cfg["inference_noise_scale_dp"] = 0.5
    assert ck.extract_voice_config(cfg, "en")["inference_noise_scale_dp"] == 0.5


def test_language_and_sample_rate() -> None:
    vc = ck.extract_voice_config(ljspeech(), "de")
    assert vc["language"] == "de" and vc["sample_rate"] == 22050


def test_coqui_keys_never_loads_pickles() -> None:
    assert "torch.load" not in (ROOT / "tools" / "convert" / "coqui_keys.py").read_text("utf-8")
