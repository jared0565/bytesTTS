# SPDX-License-Identifier: MPL-2.0
"""Sentence splitting with pysbd, as TTS-dev's synthesizer does (synthesizer.py:122).

TTS-dev always segmented as English (synthesizer.py:86). Here the voice's language is used when pysbd
knows it ("de-de" -> "de"), and English otherwise, so English voices split exactly as before.

pysbd's cost grows with the square of the text it is given (measured: 110 KB took 7-9 s, 225 KB ~35 s;
both clean modes). Text longer than WINDOW_CHARS is therefore segmented in windows. Each window ends where
pysbd itself starts the last sentence of a WINDOW_CHARS-long head, so that sentence is never cut and every
boundary is one pysbd chose. A head that is all one sentence is cut at its last whitespace, or hard-cut if
it has none. Text up to WINDOW_CHARS goes to pysbd unchanged.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

_FALLBACK = "en"
WINDOW_CHARS = 2000  # ~16 ms of pysbd per window (two passes): under 10 s per MB, linear


@functools.cache
def _segmenter(language: str, *, spans: bool) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # pysbd 0.3.4 has invalid "\s" escapes (3.12+ warns)
        import pysbd
        from pysbd.languages import LANGUAGE_CODES

    lang = language.split("-")[0].lower()
    lang = lang if lang in LANGUAGE_CODES else _FALLBACK
    if spans:  # offsets need clean=False; only used to choose where a window ends
        return pysbd.Segmenter(language=lang, clean=False, char_span=True)
    return pysbd.Segmenter(language=lang, clean=True)


def _windows(text: str, language: str) -> list[str]:
    windows, pos = [], 0
    while len(text) - pos > WINDOW_CHARS:
        head = text[pos : pos + WINDOW_CHARS]
        cut = _segmenter(language, spans=True).segment(head)[-1].start  # the last, possibly truncated, sentence
        if cut <= 0:
            cut = next((k + 1 for k in range(len(head) - 1, 0, -1) if head[k].isspace()), len(head))
        windows.append(head[:cut])
        pos += cut
    windows.append(text[pos:])
    return windows


def split_sentences(text: str, language: str) -> list[str]:
    segmenter = _segmenter(language, spans=False)
    return [str(s) for window in _windows(text, language) for s in segmenter.segment(window)]
