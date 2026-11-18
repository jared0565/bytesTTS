# SPDX-License-Identifier: MPL-2.0
"""The CSS10 Spanish and French golden traces (TTS-dev 0.22.0 in the sandbox, 2026-11-18): the text frontend.

Character voices: no espeak-ng, so these run everywhere.
"""

import pytest
from golden import load_trace

from bytestts.text.frontend import FrontendConfig, FrontendTrace, TextFrontend

VOICES = {"es-css10": ("es", "tux"), "fr-css10": ("fr", "gilles")}


@pytest.mark.parametrize("voice_id", sorted(VOICES))
def test_trace_matches_reference(voice_id: str) -> None:
    t = load_trace(voice_id)
    fe = TextFrontend(FrontendConfig.from_voice_config(t["voice_config"]))
    assert len(t["sentences"]) == 20
    for s in t["sentences"]:
        assert fe.trace(s["text"]) == FrontendTrace(s["cleaned"], None, s["token_ids"]), s["text"]


@pytest.mark.parametrize("voice_id", sorted(VOICES))
def test_trace_was_conditioned(voice_id: str) -> None:  # the reference ran with the ids Synthesizer.tts picks
    language, speaker = VOICES[voice_id]
    t = load_trace(voice_id)
    assert (t["speaker_id"], t["language_id"]) == (0, 0)
    assert (t["speakers"], t["languages"]) == ({speaker: 0}, {language: 0})
    assert t["voice_config"]["language"] == language and t["voice_config"]["text_cleaner"] == "multilingual_cleaners"
    for name in ("speaker_zeroed", "language_zeroed"):  # a zeroed embedding row moved sentence 0 past the bar
        control = t["controls"][name]
        assert control["samples"][0] != control["samples"][1] or control["max_abs"] > 1e-3, name
