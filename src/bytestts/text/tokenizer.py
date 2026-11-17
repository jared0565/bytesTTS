# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0: TTS/tts/utils/text/characters.py (vocab construction),
# TTS/tts/models/vits.py (VitsCharacters) and TTS/tts/utils/text/tokenizer.py (encode, blanks, bos/eos)
"""Phoneme text to token IDs, with the same IDs TTS-dev's tokenizer produces.

The ID order depends on the Coqui characters class named in the voice config. `VitsCharacters` lays the
vocabulary out as [pad] + punctuations + characters + [blank]; every other class uses BaseCharacters'
[pad, eos, bos, blank] + characters + punctuations. A wrong layout gives wrong IDs with no error, so an
unknown class is refused rather than guessed.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from bytestts.errors import NotSupported

_log = logging.getLogger(__name__)

_VITS_BLANK = "<BLNK>"
_VITS_CLASS = "TTS.tts.models.vits.VitsCharacters"
_IPA_CLASS = "TTS.tts.utils.text.characters.IPAPhonemes"
_BASE_CLASSES = frozenset({"TTS.tts.utils.text.characters.BaseCharacters", "TTS.tts.utils.text.characters.Graphemes"})


@dataclass(frozen=True)
class CharacterSet:
    characters: str
    punctuations: str
    pad: str | None
    eos: str | None
    bos: str | None
    blank: str | None
    is_unique: bool
    is_sorted: bool
    layout: Literal["base", "vits"] = "base"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> CharacterSet:
        """Read Coqui's `characters` config block, dispatching on its `characters_class`."""
        name = cfg.get("characters_class")
        if name == _VITS_CLASS:  # vits.py:1955: the config's eos/bos/blank are ignored
            characters = cfg["characters"] + (cfg.get("phonemes") or "")
            return cls(characters, cfg["punctuations"], cfg["pad"], None, None, _VITS_BLANK, False, True, "vits")
        if name == _IPA_CLASS or name in _BASE_CLASSES:
            characters = cfg["characters"]
            if name == _IPA_CLASS and cfg.get("phonemes") is not None:  # characters.py:401
                characters = cfg["phonemes"]
            return cls(
                characters,
                cfg["punctuations"],
                cfg["pad"],
                cfg["eos"],
                cfg["bos"],
                cfg["blank"],
                bool(cfg["is_unique"]),
                bool(cfg["is_sorted"]),
            )
        raise NotSupported(f"unsupported characters_class {name!r}: the token layout would be a guess")

    @functools.cached_property
    def vocab(self) -> list[str]:
        if self.layout == "vits":  # vits.py:1948
            return [self.pad or "", *self.punctuations, *self.characters, self.blank or ""]
        chars = list(self.characters)
        if self.is_unique:  # TTS-dev used set(), whose order is random when not sorted; first occurrence wins here
            chars = list(dict.fromkeys(chars))
        if self.is_sorted:
            chars = sorted(chars)
        specials = [s for s in (self.pad, self.eos, self.bos, self.blank) if s]
        vocab = specials + chars + list(self.punctuations)
        if self.is_unique and len(set(vocab)) != len(vocab):
            raise ValueError("duplicate characters in the character set")
        return vocab

    @functools.cached_property
    def _ids(self) -> dict[str, int]:
        return {char: idx for idx, char in enumerate(self.vocab)}  # a repeated char maps to its last index, as TTS-dev

    def char_to_id(self, char: str) -> int:
        return self._ids[char]

    def _special_id(self, special: str | None) -> int:
        return self._ids[special] if special else len(self.vocab)  # characters.py:60-75

    @property
    def blank_id(self) -> int:
        return self._special_id(self.blank)

    @property
    def bos_id(self) -> int:
        return self._special_id(self.bos)

    @property
    def eos_id(self) -> int:
        return self._special_id(self.eos)


class Tokenizer:
    """Encode phoneme text: unknown characters dropped, then blanks interspersed, then bos/eos (tokenizer.py:87)."""

    def __init__(self, chars: CharacterSet, *, add_blank: bool, use_eos_bos: bool) -> None:
        self.chars = chars
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self._warned: set[str] = set()

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for char in text:
            try:
                ids.append(self.chars.char_to_id(char))
            except KeyError:
                if char not in self._warned:  # once per character; the text itself is never logged
                    self._warned.add(char)
                    _log.warning("character %r (U+%04X) is not in the voice's vocabulary; dropped", char, ord(char))
        if self.add_blank:
            blanked = [self.chars.blank_id] * (len(ids) * 2 + 1)
            blanked[1::2] = ids
            ids = blanked
        if self.use_eos_bos:
            ids = [self.chars.bos_id, *ids, self.chars.eos_id]
        return ids
