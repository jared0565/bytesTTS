# SPDX-License-Identifier: MPL-2.0
import time

import pytest

from bytestts.text.numbers import MAX_NUMBER_DIGITS, expand_numbers_en


def test_digit_cap_is_fast_and_digitwise() -> None:
    t = time.perf_counter()
    out = expand_numbers_en("1" * 5000)
    assert time.perf_counter() - t < 1.0 and out.split()[:3] == ["one", "one", "one"]
    assert len(out.split()) == 5000


def test_long_digit_runs_scale_linearly() -> None:  # the ported decimal/ordinal regexes backtrack O(n^2)
    t = time.perf_counter()
    expand_numbers_en("1" * 40000)
    assert time.perf_counter() - t < 1.0


def test_15_digit_number_still_expands() -> None:
    assert MAX_NUMBER_DIGITS == 15
    assert "trillion" in expand_numbers_en("123456789012345")


def test_16_digits_are_spoken_digit_by_digit() -> None:
    assert (
        expand_numbers_en("1234567890123456")
        == "one two three four five six seven eight nine zero one two three four five six"
    )


@pytest.mark.parametrize(
    "text",
    [
        "$" + "9" * 5000,  # currency: int() of > 4300 digits raises ValueError in CPython 3.11+
        "$1." + "5" * 5000,
        "1," + "0" * 4999,  # comma removal joins the groups into one long run
        "1" * 5000 + "th",
        "1" * 2500 + "." + "2" * 2500,
        "-" + "7" * 5000,
        # CPython's limit counts leading zeros; the cap does not (aramid llm-review 10b577d7)
        "0" * 5000 + "1",
        "$" + "0" * 5000 + "1",
        "$1." + "0" * 5000 + "1",
        "0" * 5000 + "1st",
        "-" + "0" * 5000 + "1",
    ],
)
def test_long_runs_never_crash_and_stay_fast(text: str) -> None:
    t = time.perf_counter()
    out = expand_numbers_en(text)
    assert time.perf_counter() - t < 1.0
    assert not any(c.isdigit() for c in out)


# Reference outputs of TTS-dev 0.22.0 `english/number_norm.py`, computed 2026-11-17 by importing that
# file with inflect 7.6.0. Quirks are kept on purpose: the port keeps TTS-dev's behaviour.
@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("1937", "nineteen thirty-seven"),
        ("2000", "two thousand"),
        ("2005", "two thousand five"),
        ("1900", "nineteen hundred"),
        ("3.75", "three point seventy-five"),
        ("$5.50", "five dollars fifty cents"),
        ("£1", "one pound sterling"),
        ("1st", "first"),
        ("1,000,000", "one million"),
        ("$0.01", "one cent"),
        ("¥3", "three yen"),
        ("€5", "€five"),  # the currency regex has no €, although the table does
        ("-12", "minus twelve"),
        ("10.5.3", "ten point five.three"),
    ],
)
def test_reference_behaviour(text: str, spoken: str) -> None:
    assert expand_numbers_en(text) == spoken
