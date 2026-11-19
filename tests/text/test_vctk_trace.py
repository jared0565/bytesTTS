# SPDX-License-Identifier: MPL-2.0
"""The VCTK golden trace (TTS-dev 0.22.0 in the sandbox, 2026-11-19, speaker p280): the text frontend.

A phoneme voice, so the full trace needs espeak-ng 1.52.0, as en-ljspeech's does. Its language tag is en-gb, but its
phoneme language is Coqui's "en", which TTS-dev and BytesTTS both run as espeak-ng's en-us voice.
"""

import pytest
from golden import load_trace
from helpers import require_espeak

from bytestts.text.frontend import FrontendConfig, FrontendTrace, TextFrontend

VOICE = "en-vctk"


@pytest.mark.espeak
def test_trace_matches_reference() -> None:  # an en-gb voice still phonemizes exactly as the reference did
    t = load_trace(VOICE)
    fe = TextFrontend(FrontendConfig.from_voice_config(t["voice_config"]), espeak=require_espeak(t["espeak_version"]))
    assert len(t["sentences"]) == 20
    for s in t["sentences"]:
        assert fe.trace(s["text"]) == FrontendTrace(s["cleaned"], s["phonemes"], s["token_ids"]), s["text"]


def test_the_language_tag_is_not_the_phoneme_language() -> None:
    cfg = FrontendConfig.from_voice_config(load_trace(VOICE)["voice_config"])
    assert (cfg.kind, cfg.language, cfg.phoneme_language, cfg.cleaner) == ("espeak", "en-gb", "en", "phoneme_cleaners")


def test_trace_was_conditioned() -> None:  # the reference ran the named speaker, and other rows move its output
    t = load_trace(VOICE)
    assert (t["speaker"], t["speaker_id"], t["speakers"]["p280"]) == ("p280", 54, 54)
    assert len(t["speakers"]) == 109 and sorted(t["speakers"].values()) == list(range(109))
    assert t["speaker_other_id"] == 0 and t["language_id"] is None
    for name in ("speaker_zeroed", "speaker_other"):  # each moved sentence 0 past the parity bar
        control = t["controls"][name]
        assert control["samples"][0] != control["samples"][1] or control["max_abs"] > 1e-3, name
