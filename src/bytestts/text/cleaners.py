# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/utils/text/cleaners.py (and english/abbreviations.py)
"""Text cleaners, ported with TTS-dev's behaviour. Only the pipelines BytesTTS voices use are registered."""

from __future__ import annotations

import re
from collections.abc import Callable

from bytestts.text.numbers import expand_numbers_en

_whitespace_re = re.compile(r"\s+")

# (regular expression, replacement) pairs for English abbreviations
_abbreviations_en = [
    (re.compile(rf"\b{abbr}\.", re.IGNORECASE), spoken)
    for abbr, spoken in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]
_ABBREVIATIONS = {"en": _abbreviations_en}


def expand_abbreviations(text: str, lang: str = "en") -> str:
    for regex, replacement in _ABBREVIATIONS[lang]:
        text = regex.sub(replacement, text)
    return text


def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text).strip()


def remove_aux_symbols(text: str) -> str:
    return re.sub(r"[\<\>\(\)\[\]\"]+", "", text)


def replace_symbols(text: str, lang: str | None = "en") -> str:
    text = text.replace(";", ",")
    text = text.replace("-", " ") if lang != "ca" else text.replace("-", "")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    elif lang == "fr":
        text = text.replace("&", " et ")
    elif lang == "pt":
        text = text.replace("&", " e ")
    elif lang == "ca":
        text = text.replace("&", " i ")
        text = text.replace("'", "")
    return text


def basic_cleaners(text: str) -> str:
    """Lowercase and collapse whitespace, without transliteration."""
    return collapse_whitespace(lowercase(text))


def phoneme_cleaners(text: str) -> str:
    """For phoneme voices: numbers, abbreviations and symbols. Does not lowercase."""
    text = expand_numbers_en(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    return collapse_whitespace(text)


_CLEANERS: dict[str, Callable[[str], str]] = {
    "basic_cleaners": basic_cleaners,
    "phoneme_cleaners": phoneme_cleaners,
}


def get_cleaner(name: str) -> Callable[[str], str]:
    """The cleaner pipeline called `name`. Raises KeyError for a name BytesTTS does not ship."""
    return _CLEANERS[name]
