# SPDX-License-Identifier: MPL-2.0
import pytest
from golden import trace

from bytestts.text.punctuation import DEFAULT_PUNCTUATIONS, Punctuation, PuncPosition


def test_default_punctuations_verbatim() -> None:  # TTS-dev punctuation.py:7
    assert DEFAULT_PUNCTUATIONS == ';:,.!?¡¿—…"«»“”'


@pytest.mark.parametrize("field", ["text", "cleaned"])
def test_golden_sentences_round_trip(field: str) -> None:
    p = Punctuation()
    for s in trace["sentences"]:
        parts, puncs = p.strip_to_restore(s[field])
        assert p.restore(parts, puncs) == [s[field]], s[field]


def test_strip_to_restore_parts() -> None:
    parts, puncs = Punctuation().strip_to_restore("Hello, world!")
    assert parts == ["Hello", "world"]
    assert [(x.punc, x.position) for x in puncs] == [(", ", PuncPosition.MIDDLE), ("!", PuncPosition.END)]


def test_punctuation_only_and_no_punctuation() -> None:
    p = Punctuation()
    assert p.strip_to_restore("?!") == ([], [p.strip_to_restore("?!")[1][0]])
    assert p.restore(*p.strip_to_restore("?!")) == ["?!"]
    assert p.strip_to_restore("plain words") == (["plain words"], [])


def test_many_marks_do_not_hit_the_recursion_limit() -> None:  # TTS-dev's restore recursed once per mark
    text = "a, " * 3000 + "end."
    p = Punctuation()
    assert p.restore(*p.strip_to_restore(text)) == [text]


def test_puncs_must_be_str() -> None:
    with pytest.raises(ValueError):
        Punctuation(puncs=123)  # type: ignore[arg-type]
