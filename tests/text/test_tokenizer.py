# SPDX-License-Identifier: MPL-2.0
import logging

import pytest
from golden import VOICE_CFG, trace

from bytestts.errors import NotSupported
from bytestts.text.tokenizer import CharacterSet, Tokenizer


def _voice_tokenizer() -> Tokenizer:
    return Tokenizer(
        CharacterSet.from_config(VOICE_CFG["characters"]),
        add_blank=VOICE_CFG["add_blank"],
        use_eos_bos=VOICE_CFG["enable_eos_bos_chars"],
    )


def test_tokens_match_reference() -> None:
    tok = _voice_tokenizer()
    for s in trace["sentences"]:
        assert tok.encode(s["phonemes"]) == s["token_ids"], s["phonemes"]


def test_vits_layout() -> None:  # TTS-dev vits.py:1948: [pad] + punctuations + characters + [blank]
    cfg = VOICE_CFG["characters"]
    vocab = CharacterSet.from_config(cfg).vocab
    assert vocab[0] == cfg["pad"]
    assert vocab[1 : 1 + len(cfg["punctuations"])] == list(cfg["punctuations"])
    assert vocab[-1] == "<BLNK>"
    assert len(vocab) == 1 + len(cfg["punctuations"]) + len(cfg["characters"]) + len(cfg["phonemes"]) + 1


def test_base_layout_order() -> None:  # TTS-dev characters.py:280: [pad, eos, bos, blank] + sorted chars + puncs
    chars = CharacterSet("cba", "!.", "<PAD>", "<EOS>", "<BOS>", "<BLNK>", is_unique=False, is_sorted=True)
    assert chars.vocab == ["<PAD>", "<EOS>", "<BOS>", "<BLNK>", "a", "b", "c", "!", "."]
    unsorted = CharacterSet("cba", "!", None, "", None, None, is_unique=False, is_sorted=False)
    assert unsorted.vocab == ["c", "b", "a", "!"]  # empty or None specials are left out


def test_base_layout_from_config_reads_phonemes_for_ipa_class() -> None:  # characters.py:401, the band-aid
    block = {
        "characters_class": "TTS.tts.utils.text.characters.IPAPhonemes",
        "characters": "abc",
        "phonemes": "ə",
        "punctuations": "!",
        "pad": "<PAD>",
        "eos": None,
        "bos": None,
        "blank": "<BLNK>",
        "is_unique": False,
        "is_sorted": True,
    }
    assert CharacterSet.from_config(block).vocab == ["<PAD>", "<BLNK>", "ə", "!"]


@pytest.mark.parametrize("cls", [None, "TTS.tts.models.other.MysteryCharacters"])
def test_unknown_characters_class_is_refused(cls: str | None) -> None:  # a guessed layout gives wrong ids silently
    block = {**VOICE_CFG["characters"], "characters_class": cls}
    with pytest.raises(NotSupported):
        CharacterSet.from_config(block)


def test_unknown_chars_dropped_and_warned_once(caplog: pytest.LogCaptureFixture) -> None:
    tok = _voice_tokenizer()
    with caplog.at_level(logging.WARNING, logger="bytestts"):
        ids = tok.encode("a😀😀b")
        assert ids == tok.encode("ab")
        tok.encode("private words 😀")
    hits = [r for r in caplog.records if "😀" in r.getMessage()]
    assert len(hits) == 1 and hits[0].levelno == logging.WARNING
    assert not any("private" in r.getMessage() for r in caplog.records)  # the text itself is never logged


def test_blank_interspersed_and_bos_eos() -> None:
    chars = CharacterSet("ab", "", "<PAD>", "<EOS>", "<BOS>", "<BLNK>", is_unique=False, is_sorted=True)
    pad, eos, bos, blank, a, b = range(6)
    assert Tokenizer(chars, add_blank=True, use_eos_bos=False).encode("ab") == [blank, a, blank, b, blank]
    assert Tokenizer(chars, add_blank=False, use_eos_bos=True).encode("ab") == [bos, a, b, eos]
    assert Tokenizer(chars, add_blank=True, use_eos_bos=False).encode("") == [blank]  # as TTS-dev
    assert pad == 0
