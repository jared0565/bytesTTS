# SPDX-License-Identifier: MPL-2.0
import time
from pathlib import Path

import pytest
from golden import VOICE_CFG, trace
from helpers import CHAR_CFG, require_espeak

from bytestts.errors import NoSpeakableText, NotSupported
from bytestts.text.frontend import MAX_TOKENS_PER_CHUNK, FrontendConfig, FrontendTrace, TextFrontend
from bytestts.text.sentences import WINDOW_CHARS, split_sentences
from bytestts.text.tokenizer import CharacterSet


@pytest.mark.espeak
def test_trace_matches_reference() -> None:
    install = require_espeak(trace["espeak_version"])
    fe = TextFrontend(FrontendConfig.from_voice_config(VOICE_CFG), espeak=install)
    for s in trace["sentences"]:
        assert fe.trace(s["text"]) == FrontendTrace(s["cleaned"], s["phonemes"], s["token_ids"]), s["text"]


def test_from_voice_config_reads_the_p07_keys() -> None:
    cfg = FrontendConfig.from_voice_config(VOICE_CFG)
    assert (cfg.kind, cfg.language, cfg.phoneme_language, cfg.cleaner) == ("espeak", "en", "en", "phoneme_cleaners")
    assert (cfg.add_blank, cfg.use_eos_bos) == (True, False)
    assert cfg.characters == CharacterSet.from_config(VOICE_CFG["characters"])


@pytest.mark.parametrize("phonemizer", ["gruut", "ja_jp_phonemizer"])
def test_other_phonemizers_are_not_supported(phonemizer: str) -> None:
    with pytest.raises(NotSupported):
        FrontendConfig.from_voice_config({**VOICE_CFG, "phonemizer": phonemizer})


def test_unknown_cleaner_is_not_supported() -> None:
    with pytest.raises(NotSupported):
        TextFrontend(FrontendConfig(**{**CHAR_CFG.__dict__, "cleaner": "transliteration_cleaners_xx"}))


@pytest.mark.parametrize("text", ["", "   ", "?!...", "😀😀", "中文"])
def test_no_speakable_text(text: str) -> None:  # characters frontend: no espeak needed
    with pytest.raises(NoSpeakableText):
        TextFrontend(CHAR_CFG).encode(text)


def test_unspeakable_sentences_are_dropped_not_fatal() -> None:
    chunks = TextFrontend(CHAR_CFG).encode("Hello there. ... ?! Goodbye now.")
    assert len(chunks) == 2


def test_long_unpunctuated_text_is_chunked() -> None:
    chunks = TextFrontend(CHAR_CFG).encode(" ".join(["word"] * MAX_TOKENS_PER_CHUNK))  # >= 4x the cap in tokens
    assert len(chunks) > 1 and all(0 < len(c) <= MAX_TOKENS_PER_CHUNK for c in chunks)


def test_chunks_split_at_whitespace() -> None:
    fe = TextFrontend(CHAR_CFG)
    words = " ".join(["word"] * MAX_TOKENS_PER_CHUNK)
    for chunk in fe.encode(words):  # k whole words = 5k-1 characters = 10k-1 tokens with blanks
        assert (len(chunk) + 1) % 10 == 0, len(chunk)  # a word cut in two breaks this


def test_one_huge_word_is_still_chunked() -> None:  # no whitespace at all: halve the text
    chunks = TextFrontend(CHAR_CFG).encode("a" * 5000)
    assert len(chunks) > 1 and all(0 < len(c) <= MAX_TOKENS_PER_CHUNK for c in chunks)


def test_long_text_is_fast() -> None:  # Review Focus #3: long input must not go quadratic
    text = ("The quick brown fox jumps over the lazy dog " * 2500).strip()  # ~110 KB, one sentence
    start = time.perf_counter()
    chunks = TextFrontend(CHAR_CFG).encode(text)
    assert time.perf_counter() - start < 10
    assert all(0 < len(c) <= MAX_TOKENS_PER_CHUNK for c in chunks)


def test_characters_frontend_needs_no_espeak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("BYTESTTS_ESPEAK", raising=False)
    assert TextFrontend(CHAR_CFG).encode("hello world")


def test_split_sentences() -> None:
    assert split_sentences("Hello there. How are you?", "en") == ["Hello there.", "How are you?"]
    assert split_sentences("Hallo. Wie geht's?", "de-de") == split_sentences("Hallo. Wie geht's?", "de")
    assert split_sentences("One. Two.", "xx") == split_sentences("One. Two.", "en")  # unknown: TTS-dev's "en"


def test_sentence_split_is_linear() -> None:  # pysbd alone is quadratic: 110 KB took 7-9 s, 220 KB ~36 s
    text = ("The quick brown fox jumps over the lazy dog. " * 5000).strip()  # ~225 KB
    start = time.perf_counter()
    sentences = split_sentences(text, "en")
    assert time.perf_counter() - start < 5
    assert len(sentences) == 5000 and set(sentences) == {"The quick brown fox jumps over the lazy dog."}


def test_windowed_split_matches_pysbd_on_real_text() -> None:
    import pysbd

    corpus = (Path(__file__).resolve().parents[1] / "corpus" / "en.txt").read_text("utf-8")
    text = " ".join([corpus.replace("\n", " ")] * 8)  # several windows, sentence ends at the cuts
    assert len(text) > 3 * WINDOW_CHARS
    assert split_sentences(text, "en") == pysbd.Segmenter(language="en", clean=True).segment(text)
