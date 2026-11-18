# SPDX-License-Identifier: MPL-2.0
"""convert_coqui.convert_state: everything after the unpickle, on an in-memory state dict (needs torch).

The unpickle itself (zip, picklescan, torch.load) runs only in the sandbox; see test_convert_coqui.py.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from bytestts.train.vits.io import load_generator

ROOT = Path(__file__).resolve().parents[2]


def load_tool(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cc = load_tool("tools/convert/convert_coqui.py")
fixture_tool = load_tool("tools/dev/make_tiny_fixture.py")
MODEL_ARGS = {**fixture_tool.TINY_ARGS, "use_sdp": False}
MULTI_ARGS = {**MODEL_ARGS, "use_speaker_embedding": True, "num_speakers": 3, "speaker_embedding_channels": 8}


def legacy(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The port's parametrize names back to the weight_g/weight_v names Coqui's releases carry."""
    names = {".parametrizations.weight.original0": ".weight_g", ".parametrizations.weight.original1": ".weight_v"}
    out = {}
    for key, value in sd.items():
        for new, old in names.items():
            if key.endswith(new):
                key = key[: -len(new)] + old
        out[key] = value
    return out


@pytest.fixture(scope="module")
def tiny() -> Any:
    return fixture_tool._tiny_model(MODEL_ARGS)


def checkpoint(model: Any, **extra: torch.Tensor) -> dict[str, torch.Tensor]:
    return {**legacy(model.state_dict()), **extra}


def convert(state: dict[str, torch.Tensor], out: Path, *, model_args: dict[str, Any] = MODEL_ARGS, **kw: Any) -> Any:
    cfg = fixture_tool.coqui_config(model_args)
    for change in kw.pop("config_changes", []):
        change(cfg)
    return cc.convert_state(state, cfg, language="en", speaker_ids=kw.pop("speaker_ids", None), out=out)


def refused(code: int, match: str, call: Any) -> None:
    with pytest.raises(cc.Refused, match=match) as info:
        call()
    assert info.value.code == code


def test_round_trip(tiny: Any, tmp_path: Path) -> None:
    report = convert(checkpoint(tiny), tmp_path / "c")
    restored = load_generator(tmp_path / "c").state_dict()
    original = tiny.state_dict()
    assert restored.keys() == original.keys()
    assert all(torch.equal(restored[k], original[k]) for k in original)
    assert {p.name for p in (tmp_path / "c").iterdir()} == {
        "generator.safetensors",
        "model_config.json",
        "voice_config.json",
        "conversion_report.json",
    }
    assert report["num_chars"] == tiny.cfg.num_chars and report["discriminator_keys"] == 0
    assert json.loads((tmp_path / "c" / "conversion_report.json").read_bytes()) == report


def test_discriminator_is_kept_apart(tiny: Any, tmp_path: Path) -> None:
    from safetensors.torch import load_file

    w = torch.ones(2, 1, 3)
    state = checkpoint(tiny, **{"disc.a.weight_g": w[:, :, :1], "disc.a.weight_v": w, "disc.a.bias": w[:, 0, 0]})
    report = convert(state, tmp_path / "c")
    disc = load_file(str(tmp_path / "c" / "discriminator.safetensors"))
    assert set(disc) == {"a.parametrizations.weight.original0", "a.parametrizations.weight.original1", "a.bias"}
    assert report["discriminator_keys"] == 3


def test_resampler_kernel_is_dropped_and_reported(tiny: Any, tmp_path: Path) -> None:
    report = convert(checkpoint(tiny, **{"audio_resampler.kernel": torch.zeros(3)}), tmp_path / "c")
    assert report["dropped_keys"] == ["audio_resampler.kernel"]


def test_shape_mismatch_is_refused(tiny: Any, tmp_path: Path) -> None:
    change = [lambda c: c["model_args"].__setitem__("num_layers_flow", 2)]
    refused(4, "does not fit", lambda: convert(checkpoint(tiny), tmp_path / "c", config_changes=change))


def test_missing_key_is_refused(tiny: Any, tmp_path: Path) -> None:
    state = checkpoint(tiny)
    state.pop("text_encoder.proj.bias")
    refused(4, "does not fit", lambda: convert(state, tmp_path / "c"))


def test_unexpected_key_is_refused(tiny: Any, tmp_path: Path) -> None:
    refused(4, "does not fit", lambda: convert(checkpoint(tiny, **{"flow.extra": torch.zeros(1)}), tmp_path / "c"))


def test_vocabulary_must_fit_the_embedding(tiny: Any, tmp_path: Path) -> None:
    change = [lambda c: c["characters"].__setitem__("characters", "abc")]
    refused(4, "vocabulary", lambda: convert(checkpoint(tiny), tmp_path / "c", config_changes=change))


def test_unsupported_voice_kinds_are_refused(tiny: Any, tmp_path: Path) -> None:
    change = [lambda c: c["model_args"].__setitem__("use_d_vector_file", True)]
    refused(5, "not supported", lambda: convert(checkpoint(tiny), tmp_path / "c", config_changes=change))


def test_refusals_write_nothing(tiny: Any, tmp_path: Path) -> None:
    change = [lambda c: c["characters"].__setitem__("characters", "abc")]
    refused(4, "vocabulary", lambda: convert(checkpoint(tiny), tmp_path / "c", config_changes=change))
    assert not (tmp_path / "c").exists()


def test_multi_speaker_needs_speaker_ids(tmp_path: Path) -> None:
    model = fixture_tool._tiny_model(MULTI_ARGS)
    refused(4, "speaker_ids", lambda: convert(checkpoint(model), tmp_path / "c", model_args=MULTI_ARGS))


def test_multi_speaker_writes_speakers(tmp_path: Path) -> None:
    model = fixture_tool._tiny_model(MULTI_ARGS)
    ids = {"alice": 0, "bob": 2, "carol": 1}
    report = convert(checkpoint(model), tmp_path / "c", model_args=MULTI_ARGS, speaker_ids=ids)
    assert json.loads((tmp_path / "c" / "speakers.json").read_bytes()) == ids and report["num_speakers"] == 3


def test_coqui_config_is_independent_of_the_fixture_constants() -> None:  # a shared dict once leaked between tests
    before = json.dumps(fixture_tool.VOICE_CONFIG, sort_keys=True)
    cfg = fixture_tool.coqui_config(MODEL_ARGS)
    cfg["characters"]["characters"] = "abc"
    cfg["model_args"]["resblock_kernel_sizes_decoder"].append(99)
    assert json.dumps(fixture_tool.VOICE_CONFIG, sort_keys=True) == before
    assert 99 not in MODEL_ARGS["resblock_kernel_sizes_decoder"]


# The CSS10 layout: a one-row speaker table and a one-row language table. The configs say num_speakers 0;
# TTS-dev takes the count from speaker_ids.json, and the language from language_ids.json.
CSS10_ARGS = {
    **MODEL_ARGS,
    "use_language_embedding": True,
    "num_languages": 1,
    "embedded_language_dim": 4,
    "use_speaker_embedding": True,
    "num_speakers": 1,
    "speaker_embedding_channels": 8,
}


@pytest.fixture(scope="module")
def css10() -> Any:
    return fixture_tool._tiny_model(CSS10_ARGS)


def convert_css10(model: Any, out: Path, **kw: Any) -> Any:
    cfg = fixture_tool.coqui_config({**CSS10_ARGS, "num_speakers": 0})  # as the CSS10 configs say
    return cc.convert_state(
        checkpoint(model),
        cfg,
        language="es",
        speaker_ids=kw.get("speaker_ids", {"tux": 0}),
        language_ids=kw.get("language_ids", {"es": 0}),
        out=out,
    )


def test_language_voice_converts(css10: Any, tmp_path: Path) -> None:
    report = convert_css10(css10, tmp_path / "c")
    assert report["num_speakers"] == 1 and report["languages"] == ["es"]
    port = json.loads((tmp_path / "c" / "model_config.json").read_bytes())
    assert (port["num_speakers"], port["num_languages"], port["use_language_embedding"]) == (1, 1, True)
    voice = json.loads((tmp_path / "c" / "voice_config.json").read_bytes())
    assert voice["model_args"]["num_speakers"] == 0  # Coqui's own model_args, unchanged
    assert json.loads((tmp_path / "c" / "speakers.json").read_bytes()) == {"tux": 0}
    loaded = load_generator(tmp_path / "c")
    assert torch.equal(loaded.emb_l.weight, css10.emb_l.weight) and loaded.emb_g.weight.shape == (1, 8)


def test_language_voice_needs_language_ids(css10: Any, tmp_path: Path) -> None:
    refused(4, "language_ids", lambda: convert_css10(css10, tmp_path / "c", language_ids=None))
    assert not (tmp_path / "c").exists()


def test_language_ids_must_name_the_voice_language(css10: Any, tmp_path: Path) -> None:
    refused(4, "'fr'", lambda: convert_css10(css10, tmp_path / "c", language_ids={"fr": 0}))
    refused(4, "'es'", lambda: convert_css10(css10, tmp_path / "c", language_ids={"es": 1}))


def test_multi_language_tables_are_refused(css10: Any, tmp_path: Path) -> None:
    refused(5, "language", lambda: convert_css10(css10, tmp_path / "c", language_ids={"es": 0, "fr": 1}))


def test_language_ids_without_a_language_embedding_are_ignored(tiny: Any, tmp_path: Path) -> None:
    cfg = fixture_tool.coqui_config(MODEL_ARGS)
    report = cc.convert_state(
        checkpoint(tiny), cfg, language="en", speaker_ids=None, language_ids={"en": 0}, out=tmp_path / "c"
    )
    assert report["languages"] == [] and not hasattr(load_generator(tmp_path / "c"), "emb_l")
