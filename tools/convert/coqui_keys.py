# SPDX-License-Identifier: MPL-2.0
"""Pure helpers for converting Coqui VITS checkpoints: state-dict keys and the voice config.

No pickles are opened here (that happens only in convert_coqui.py, inside the sandbox), so this module is safe to
import and test on the host.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from bytestts.train.vits.config import VitsModelConfig

V = TypeVar("V")

DISC = "disc."
# Not part of the inference generator: Coqui's speaker encoder (for d-vectors) and torchaudio's Resample kernel
# buffer, which the reference implementation keeps under audio_resampler when encoder_sample_rate is set.
DROPPED = ("speaker_encoder.", "audio_resampler.")
_G, _V = ".weight_g", ".weight_v"
_ORIGINAL = {_G: ".parametrizations.weight.original0", _V: ".parametrizations.weight.original1"}

# The P0.7 trace.json voice_config keys, in its order.
VOICE_CONFIG_KEYS = (
    "language",
    "use_phonemes",
    "phonemizer",
    "phoneme_language",
    "text_cleaner",
    "characters",
    "add_blank",
    "enable_eos_bos_chars",
    "sample_rate",
    "inference_noise_scale",
    "length_scale",
    "inference_noise_scale_dp",
)
_SCALES = ("inference_noise_scale", "length_scale", "inference_noise_scale_dp")


def split_state_dict(sd: Mapping[str, V]) -> tuple[dict[str, V], dict[str, V]]:
    """(generator, discriminator with its `disc.` prefix stripped); the speaker encoder and resampler dropped."""
    generator: dict[str, V] = {}
    discriminator: dict[str, V] = {}
    for key, value in sd.items():
        if key.startswith(DISC):
            discriminator[key[len(DISC) :]] = value
        elif not key.startswith(DROPPED):
            generator[key] = value
    return generator, discriminator


def rename_legacy_weight_norm(sd: Mapping[str, V]) -> dict[str, V]:
    """`X.weight_g`/`X.weight_v` (torch.nn.utils.weight_norm) -> the parametrize names the port uses."""
    out: dict[str, V] = {}
    for key, value in sd.items():
        suffix = next((s for s in _ORIGINAL if key.endswith(s)), None)
        if suffix is not None:
            base = key[: -len(suffix)]
            other = _V if suffix == _G else _G
            if base + other not in sd:
                raise ValueError(f"{key} has no matching {base + other}: half a weight norm")
            key = base + _ORIGINAL[suffix]
        if key in out or (suffix is not None and key in sd):
            raise ValueError(f"two keys map to {key}")
        out[key] = value
    return out


def extract_voice_config(coqui_cfg: Mapping[str, Any], language: str) -> dict[str, Any]:
    """The P0.7 voice_config, plus `model_args`: the part of Coqui's model_args the generator reads.

    The synthesis scales come from the top level when present, else from model_args (v0.6.1 configs keep them
    only there), exactly as Phase 0 read them.
    """
    model_args: Mapping[str, Any] = coqui_cfg.get("model_args") or {}
    voice: dict[str, Any] = {
        "language": language,
        "use_phonemes": coqui_cfg.get("use_phonemes"),
        "phonemizer": coqui_cfg.get("phonemizer"),
        "phoneme_language": coqui_cfg.get("phoneme_language"),
        "text_cleaner": coqui_cfg.get("text_cleaner"),
        "characters": coqui_cfg.get("characters"),
        "add_blank": coqui_cfg.get("add_blank"),
        "enable_eos_bos_chars": coqui_cfg.get("enable_eos_bos_chars"),
        "sample_rate": coqui_cfg["audio"]["sample_rate"],
    }
    for key in _SCALES:
        voice[key] = coqui_cfg[key] if key in coqui_cfg else model_args.get(key)
    if tuple(voice) != VOICE_CONFIG_KEYS:  # the P0.7 schema, key for key
        raise RuntimeError(f"voice_config keys {tuple(voice)} drifted from {VOICE_CONFIG_KEYS}")
    voice["model_args"] = {k: v for k, v in model_args.items() if k in VitsModelConfig.coqui_keys()}
    return voice
