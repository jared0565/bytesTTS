# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/utils/text/punctuation.py
"""Strip punctuation before phonemizing and restore it afterwards (TTS-dev behaviour).

Two changes, neither visible in results: `restore` is iterative (TTS-dev recursed once per mark,
so about 1,000 marks in one sentence raised RecursionError), and `six` is not needed.
"""

from __future__ import annotations

import re
from collections import deque
from enum import Enum
from typing import NamedTuple

DEFAULT_PUNCTUATIONS = ';:,.!?¡¿—…"«»“”'


class PuncPosition(Enum):
    """Where a run of punctuation sat in the text."""

    BEGIN = 0
    END = 1
    MIDDLE = 2


class _PuncIndex(NamedTuple):
    punc: str
    position: PuncPosition


class Punctuation:
    """Strip punctuation from text, or strip it and restore it later."""

    def __init__(self, puncs: str = DEFAULT_PUNCTUATIONS) -> None:
        self.puncs = puncs

    @staticmethod
    def default_puncs() -> str:
        return DEFAULT_PUNCTUATIONS

    @property
    def puncs(self) -> str:
        return self._puncs

    @puncs.setter
    def puncs(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("[!] Punctuations must be of type str.")
        self._puncs = "".join(dict.fromkeys(value))  # remove duplicates, keep the order
        self.puncs_regular_exp = re.compile(rf"(\s*[{re.escape(self._puncs)}]+\s*)+")

    def strip(self, text: str) -> str:
        """Replace every run of punctuation with a space, then trim."""
        return self.puncs_regular_exp.sub(" ", text).strip()

    def strip_to_restore(self, text: str) -> tuple[list[str], list[_PuncIndex]]:
        """'This is. example !' -> (['This is', 'example'], [('. ', MIDDLE), (' !', END)])."""
        matches = list(self.puncs_regular_exp.finditer(text))
        if not matches:
            return [text], []
        if len(matches) == 1 and matches[0].group() == text:  # the text is only punctuation
            return [], [_PuncIndex(text, PuncPosition.BEGIN)]
        puncs = []
        for match in matches:
            position = PuncPosition.MIDDLE
            if match is matches[0] and text.startswith(match.group()):
                position = PuncPosition.BEGIN
            elif match is matches[-1] and text.endswith(match.group()):
                position = PuncPosition.END
            puncs.append(_PuncIndex(match.group(), position))
        splitted_text = []
        for idx, punc in enumerate(puncs):
            split = text.split(punc.punc)
            prefix, suffix = split[0], punc.punc.join(split[1:])
            text = suffix
            if prefix == "":  # no empty item for an initial punctuation
                continue
            splitted_text.append(prefix)
            if idx == len(puncs) - 1 and len(suffix) > 0:  # text after the last punctuation
                splitted_text.append(suffix)
        return splitted_text, puncs

    @classmethod
    def restore(cls, text: list[str], puncs: list[_PuncIndex]) -> list[str]:
        """(['This is', 'example'], [('. ', MIDDLE), ('!', END)]) -> ['This is. example!']."""
        queue = deque(text)
        out: list[str] = []
        for k, current in enumerate(puncs):
            if not queue:  # nothing was phonemized: the remaining marks alone
                out.append("".join(p.punc for p in puncs[k:]))
                return out
            if current.position == PuncPosition.BEGIN:
                queue[0] = current.punc + queue[0]
            elif current.position == PuncPosition.END:
                out.append(queue.popleft() + current.punc)
            elif len(queue) == 1:  # the part after a middle mark was not phonemized
                queue[0] = queue[0] + current.punc
            else:
                first = queue.popleft()
                queue[0] = first + current.punc + queue[0]
        return out + list(queue)
