# SPDX-License-Identifier: MPL-2.0
import pytest
from golden import VOICE_CFG, trace

from bytestts.text.cleaners import get_cleaner


def test_cleaner_matches_reference() -> None:
    clean = get_cleaner(VOICE_CFG["text_cleaner"])
    for item in trace["sentences"]:
        assert clean(item["text"]) == item["cleaned"], item["text"]


def test_basic_cleaners() -> None:  # TTS-dev cleaners.py:79-83; collapse_whitespace strips (L34)
    assert get_cleaner("basic_cleaners")("  Hello   World ") == "hello world"


def test_phoneme_cleaners_does_not_lowercase() -> None:
    assert get_cleaner("phoneme_cleaners")("Hello (World)") == "Hello World"


def test_unknown_cleaner_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_cleaner("no_such_cleaners")
