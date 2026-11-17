# SPDX-License-Identifier: MPL-2.0
"""Text to model input: sentences -> cleaner -> (espeak-ng phonemes) -> token IDs, in bounded chunks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from bytestts.errors import NoSpeakableText, NotSupported
from bytestts.text.cleaners import get_cleaner
from bytestts.text.phonemizers.espeak import EspeakInstall, EspeakPhonemizer
from bytestts.text.sentences import split_sentences
from bytestts.text.tokenizer import CharacterSet, Tokenizer

# PROVISIONAL (P0.9, operator 2026-11-17). Measured at 1 thread: 500 real-text tokens peaked at 500 MiB RAM
# (phase0-findings §6). The final value comes from the deferred latency re-run,
# `onnx_check.py --chunk-cap --real-ids`, at a representative thread count once the hardware is cleared.
MAX_TOKENS_PER_CHUNK = 500


def _speakable(cleaned: str) -> bool:
    return any(c.isalnum() for c in cleaned)


@dataclass(frozen=True)
class FrontendConfig:
    kind: Literal["espeak", "characters"]
    language: str
    phoneme_language: str | None
    cleaner: str
    characters: CharacterSet
    add_blank: bool
    use_eos_bos: bool

    @classmethod
    def from_voice_config(cls, cfg: Mapping[str, Any]) -> FrontendConfig:
        """Read the P0.7 `voice_config` keys."""
        use_phonemes = bool(cfg["use_phonemes"])
        phonemizer = cfg.get("phonemizer") or "espeak"  # TTS-dev's default for en/de/es/fr/it is espeak
        if use_phonemes and phonemizer != "espeak":
            raise NotSupported(f"phonemizer {phonemizer!r} is not supported; only espeak-ng is")
        phoneme_language = cfg.get("phoneme_language")
        if use_phonemes and not phoneme_language:
            raise ValueError("a phoneme voice needs phoneme_language")
        return cls(
            kind="espeak" if use_phonemes else "characters",
            language=str(cfg["language"]),
            phoneme_language=str(phoneme_language) if use_phonemes else None,
            cleaner=str(cfg["text_cleaner"]),
            characters=CharacterSet.from_config(cfg["characters"]),
            add_blank=bool(cfg["add_blank"]),
            use_eos_bos=bool(cfg["enable_eos_bos_chars"]),
        )


@dataclass(frozen=True)
class FrontendTrace:
    cleaned: str
    phonemes: str | None
    token_ids: list[int]


def _split_near_middle(text: str) -> tuple[str, str]:
    """Split at the whitespace nearest the middle; a text with none is halved."""
    middle = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c.isspace()]
    if not spaces:
        return text[:middle], text[middle:]
    i = min(spaces, key=lambda k: abs(k - middle))
    return text[:i], text[i + 1 :]


class TextFrontend:
    def __init__(self, cfg: FrontendConfig, *, espeak: EspeakInstall | None = None) -> None:
        self.cfg = cfg
        try:
            self._clean = get_cleaner(cfg.cleaner)
        except KeyError as e:
            raise NotSupported(f"text cleaner {cfg.cleaner!r} is not supported") from e
        self._phonemizer = None
        if cfg.kind == "espeak":
            self._phonemizer = EspeakPhonemizer(cfg.phoneme_language or cfg.language, install=espeak)
        self._tokenizer = Tokenizer(cfg.characters, add_blank=cfg.add_blank, use_eos_bos=cfg.use_eos_bos)
        self._specials = {cfg.characters.blank_id, cfg.characters.bos_id, cfg.characters.eos_id}

    def _symbols(self, cleaned: str) -> str | None:
        return None if self._phonemizer is None else self._phonemizer.phonemize(cleaned, separator="")

    def trace(self, sentence: str) -> FrontendTrace:
        """One sentence, unchunked: what TTS-dev's text_to_ids does, step by step."""
        cleaned = self._clean(sentence)
        phonemes = self._symbols(cleaned)
        return FrontendTrace(cleaned, phonemes, self._tokenizer.encode(cleaned if phonemes is None else phonemes))

    def _chunks(self, cleaned: str) -> list[list[int]]:
        if not _speakable(cleaned):
            return []
        phonemes = self._symbols(cleaned)
        ids = self._tokenizer.encode(cleaned if phonemes is None else phonemes)
        if all(i in self._specials for i in ids):  # e.g. letters with no symbol in this voice
            return []
        if len(ids) <= MAX_TOKENS_PER_CHUNK:
            return [ids]
        left, right = _split_near_middle(cleaned)
        return self._chunks(left) + self._chunks(right)

    def encode(self, text: str) -> list[list[int]]:
        """One token list per sentence, or per chunk of a sentence longer than MAX_TOKENS_PER_CHUNK."""
        sentences = split_sentences(text, self.cfg.language)
        chunks = [c for sentence in sentences for c in self._chunks(self._clean(sentence))]
        if not chunks:
            raise NoSpeakableText("the text has nothing this voice can speak")
        return chunks
