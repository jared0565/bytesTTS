# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/utils/text/english/number_norm.py
"""English number normalisation (originally from keithito/tacotron).

Behaviour is TTS-dev's, quirks included (for example "€5" -> "€five": the currency regex lacks €).
The one change: a digit run longer than MAX_NUMBER_DIGITS is spoken digit by digit. That keeps
inflect off huge numbers and every int() under CPython's 4,300-digit conversion limit.
"""

from __future__ import annotations

import functools
import re
from typing import Any

MAX_NUMBER_DIGITS = 15
_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
# (?<![0-9]) is the one change to TTS-dev's patterns. Without it, a long digit run with no "." or
# suffix after it is retried from every position inside the run, which is O(n^2). The matches are
# unchanged: a match starting inside a run implies one starting at the run's first digit, which
# re.sub finds first, and every match ends where a digit run ends.
_decimal_number_re = re.compile(r"((?<![0-9])[0-9]+\.[0-9]+)")
_currency_re = re.compile(r"(£|\$|¥)([0-9\,\.]*[0-9]+)")
_ordinal_re = re.compile(r"(?<![0-9])[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"-?[0-9]+")

_CURRENCIES: dict[str, dict[float, str]] = {
    "$": {0.01: "cent", 0.02: "cents", 1: "dollar", 2: "dollars"},
    "€": {0.01: "cent", 0.02: "cents", 1: "euro", 2: "euros"},
    "£": {0.01: "penny", 0.02: "pence", 1: "pound sterling", 2: "pounds sterling"},
    "¥": {0.02: "sen", 2: "yen"},
}


@functools.cache
def _engine() -> Any:
    import inflect  # deferred: no import-time work in bytestts (plan, Global Constraints)

    return inflect.engine()


def _spell_digits(digits: str) -> str:
    return " ".join(_DIGIT_WORDS[int(c)] for c in digits)


def _too_long(digits: str) -> bool:
    return len(digits.lstrip("0")) > MAX_NUMBER_DIGITS


def _remove_commas(m: re.Match[str]) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match[str]) -> str:
    return m.group(1).replace(".", " point ")


def _currency_amount(digits: str) -> tuple[bool, int | None, str]:
    """(is non-zero, value if small enough to convert, the text TTS-dev would print for it)."""
    if not digits:
        return False, 0, "0"
    if _too_long(digits):
        return True, None, digits.lstrip("0")  # huge: plural unit; _number_re spells the digits later
    value = int(digits)
    return value > 0, value, str(value)


def _expand_currency_value(value: str, inflection: dict[float, str]) -> str:
    parts = value.replace(",", "").split(".")
    if len(parts) > 2:
        return f"{value} {inflection[2]}"  # unexpected format
    text = []
    nonzero, integer, integer_text = _currency_amount(parts[0])
    if nonzero:
        unit = inflection.get(integer, inflection[2]) if integer is not None else inflection[2]
        text.append(f"{integer_text} {unit}")
    nonzero, fraction, fraction_text = _currency_amount(parts[1] if len(parts) > 1 else "")
    if nonzero:
        unit = inflection.get(fraction / 100, inflection[0.02]) if fraction is not None else inflection[0.02]
        text.append(f"{fraction_text} {unit}")
    if not text:
        return f"zero {inflection[2]}"
    return " ".join(text)


def _expand_currency(m: re.Match[str]) -> str:
    return _expand_currency_value(m.group(2), _CURRENCIES[m.group(1)])


def _expand_ordinal(m: re.Match[str]) -> str:
    digits = m.group(0)[:-2]
    if _too_long(digits):
        return f"{_spell_digits(digits)} {m.group(1)}"
    return str(_engine().number_to_words(m.group(0)))


def _expand_number(m: re.Match[str]) -> str:
    token = m.group(0)
    digits = token.lstrip("-")
    if _too_long(digits):
        return ("minus " if token.startswith("-") else "") + _spell_digits(digits)
    num = int(token)
    p = _engine()
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        if 2000 < num < 2010:
            return "two thousand " + str(p.number_to_words(num % 100))
        if num % 100 == 0:
            return str(p.number_to_words(num // 100)) + " hundred"
        return str(p.number_to_words(num, andword="", zero="oh", group=2)).replace(", ", " ")
    return str(p.number_to_words(num, andword=""))


def expand_numbers_en(text: str) -> str:
    text = _comma_number_re.sub(_remove_commas, text)
    text = _currency_re.sub(_expand_currency, text)
    text = _decimal_number_re.sub(_expand_decimal_point, text)
    text = _ordinal_re.sub(_expand_ordinal, text)
    return _number_re.sub(_expand_number, text)
