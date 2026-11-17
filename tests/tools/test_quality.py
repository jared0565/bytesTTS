# SPDX-License-Identifier: MPL-2.0
"""tools/dev/quality.py: the scoring rules approved before any measurement (RTF and corpus-level WER)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from bytestts.audio import Audio

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "corpus" / "en.txt"
THRESHOLDS = ROOT / "tests" / "quality" / "thresholds.json"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("quality", ROOT / "tools" / "dev" / "quality.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q = _load()


# --- normalisation: NFC, casefold, every Unicode punctuation character to a space, whitespace collapsed ---


def test_normalise_drops_punctuation_and_case() -> None:
    assert q.normalise("Dr. Smith,  said: “Hi!”") == ["dr", "smith", "said", "hi"]


def test_normalise_splits_on_hyphens_and_apostrophes() -> None:  # both are Unicode punctuation (Pd, Po)
    assert q.normalise("thirty-seven O'Brien") == ["thirty", "seven", "o", "brien"]


def test_normalise_composes_then_casefolds() -> None:
    assert q.normalise("Café STRASSE Straße") == ["café", "strasse", "strasse"]


def test_normalise_keeps_digits_and_decimal_points_split() -> None:
    assert q.normalise("3.75 inches") == ["3", "75", "inches"]


# --- word edit distance ---


@pytest.mark.parametrize(
    ("ref", "hyp", "edits"),
    [
        (["a", "b", "c"], ["a", "b", "c"], 0),
        (["a", "b", "c"], ["a", "x", "c"], 1),
        (["a", "b", "c"], ["a", "c"], 1),
        (["a", "c"], ["a", "b", "c"], 1),
        (["a", "b", "c"], [], 3),
        ([], ["a", "b"], 2),
        (["a", "b", "c", "d"], ["b", "c", "d", "e"], 2),
    ],
)
def test_word_edits(ref: list[str], hyp: list[str], edits: int) -> None:
    assert q.word_edits(ref, hyp) == edits


# --- corpus WER: per line the reference (written or spoken) with fewer edits, then total edits / total words ---


def test_wer_takes_the_spoken_reference_when_it_fits_better() -> None:
    assert q.corpus_wer([("in 1937", "in nineteen thirty-seven")], ["In nineteen thirty seven."]) == 0.0


def test_wer_takes_the_written_reference_when_it_fits_better() -> None:
    assert q.corpus_wer([("in 1937", "in nineteen thirty-seven")], ["in 1937"]) == 0.0


def test_wer_is_corpus_level_not_a_mean_of_line_rates() -> None:
    refs = [("a b c d", "a b c d"), ("e f g h i j", "e f g h i j")]
    assert q.corpus_wer(refs, ["a b c x", "e f g h i j"]) == pytest.approx(1 / 10)  # a mean of rates gives 0.125


def test_wer_counts_the_chosen_references_words() -> None:  # written: 1 edit / 4 words; spoken: 2 edits / 2
    assert q.corpus_wer([("a b c d", "a b")], ["a b c x"]) == pytest.approx(1 / 4)


def test_wer_breaks_a_tie_on_the_written_reference() -> None:  # 1 edit either way: written 1/2, spoken 1/4
    assert q.corpus_wer([("a b", "a b c d")], ["a b c"]) == pytest.approx(1 / 2)


def test_wer_needs_one_hypothesis_per_line() -> None:
    with pytest.raises(ValueError, match="hypotheses"):
        q.corpus_wer([("a", "a"), ("b", "b")], ["a"])


# --- the corpus: the written file and its .spoken.txt sibling, line for line ---


def test_shipped_english_corpus_loads_twenty_pairs() -> None:
    pairs = q.load_corpus(CORPUS)
    assert len(pairs) == 20
    differing = [i for i, (written, spoken) in enumerate(pairs) if written != spoken]
    assert differing == [0, 1, 2, 3, 4]  # the number and abbreviation lines only (P0.1)


def test_corpus_language_is_the_file_stem() -> None:
    assert q.corpus_language(CORPUS) == "en"


def test_corpus_with_a_different_line_count_is_refused(tmp_path: Path) -> None:
    (tmp_path / "en.txt").write_bytes(b"one\ntwo\n")
    (tmp_path / "en.spoken.txt").write_bytes(b"one\n")
    with pytest.raises(ValueError, match="lines"):
        q.load_corpus(tmp_path / "en.txt")


def test_corpus_with_a_blank_line_is_refused(tmp_path: Path) -> None:
    (tmp_path / "en.txt").write_bytes(b"one\n\ntwo\n")
    (tmp_path / "en.spoken.txt").write_bytes(b"one\n\ntwo\n")
    with pytest.raises(ValueError, match="blank"):
        q.load_corpus(tmp_path / "en.txt")


# --- RTF: the median over runs of each run's (synthesis seconds / audio seconds) ---


def _audio(seconds: float) -> Audio:
    return Audio(np.zeros(int(seconds * 100), dtype=np.float32), 100)


def test_rtf_is_the_median_of_each_runs_own_ratio() -> None:
    # ratios 0.1, 0.6, 0.1: median 0.1. Their mean is 0.27; the times over a fixed 10 s give a median of 0.2
    runs = [q.Run(1.0, [_audio(10.0)]), q.Run(3.0, [_audio(2.0), _audio(3.0)]), q.Run(2.0, [_audio(20.0)])]
    assert q.rtf(runs) == pytest.approx(0.1)
    assert q.rtf(runs[1:]) == pytest.approx(0.35)  # an even count: the mean of the middle two


def test_rtf_refuses_no_runs() -> None:
    with pytest.raises(ValueError):
        q.rtf([])


def test_rtf_refuses_a_run_without_audio() -> None:
    with pytest.raises(ValueError):
        q.rtf([q.Run(1.0, [])])


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_measure_times_each_run_after_an_untimed_warm_up() -> None:
    clock, calls = _Clock(), []

    def synthesize(text: str) -> Audio:
        calls.append(text)
        clock.now += len(text)  # each call "takes" as many seconds as the text has characters
        return Audio(np.zeros(100, dtype=np.float32), 100)

    runs = q.measure(synthesize, ["ab", "cde"], runs=3, clock=clock)
    assert calls == ["ab"] + ["ab", "cde"] * 3  # one warm-up sentence, then three full passes
    assert [r.seconds for r in runs] == [5.0, 5.0, 5.0]  # the warm-up's 2 s are not counted
    assert all([a.duration for a in r.audio] == [1.0, 1.0] for r in runs)


# --- resampling for the ASR model, which takes 16 kHz audio ---


def test_resample_keeps_a_tone_at_its_frequency() -> None:
    rate, seconds, hz = 22050, 1.0, 440.0
    t = np.arange(int(rate * seconds)) / rate
    out = q.resample(np.sin(2 * np.pi * hz * t).astype(np.float32), rate, 16000)
    assert out.dtype == np.float32 and len(out) == 16000
    spectrum = np.abs(np.fft.rfft(out))
    assert np.argmax(spectrum) * 16000 / len(out) == pytest.approx(hz, abs=1.0)
    assert float(np.max(np.abs(out[100:-100]))) == pytest.approx(1.0, abs=0.02)


def test_resample_at_the_same_rate_is_unchanged() -> None:
    x = np.linspace(-1, 1, 50, dtype=np.float32)
    assert np.array_equal(q.resample(x, 16000, 16000), x)


# --- thresholds: the values the operator approved on 2026-11-16, before any measurement ---


def test_thresholds_file_holds_the_approved_values() -> None:
    t = json.loads(THRESHOLDS.read_bytes())
    assert t["rtf"] == {"cpu": 0.30, "cuda": 0.05, "ci-cpu": 1.0}
    assert t["wer"] == {"en": 0.10, "de": 0.15, "es": 0.15, "fr": 0.15, "it": 0.15}
    assert t["asr"] == {
        "package": "faster-whisper==1.2.1",
        "model": "Systran/faster-whisper-large-v3",
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        "compute_type": "int8",
        "beam_size": 5,
    }


def _thresholds() -> dict[str, dict[str, float]]:
    return {"rtf": {"cpu": 0.30, "cuda": 0.05, "ci-cpu": 1.0}, "wer": {"en": 0.10}}


def test_failures_empty_at_the_bars() -> None:  # the bars are inclusive: <=
    assert q.failures(rtf=0.30, wer=0.10, rtf_key="cpu", language="en", thresholds=_thresholds()) == []


def test_failures_name_each_breach() -> None:
    found = q.failures(rtf=0.31, wer=0.1001, rtf_key="cpu", language="en", thresholds=_thresholds())
    assert len(found) == 2 and "RTF" in found[0] and "WER" in found[1]


def test_failures_use_the_requested_rtf_bar() -> None:
    assert q.failures(rtf=0.9, wer=0.0, rtf_key="ci-cpu", language="en", thresholds=_thresholds()) == []
    assert q.failures(rtf=0.06, wer=0.0, rtf_key="cuda", language="en", thresholds=_thresholds()) != []


@pytest.mark.parametrize(
    "extra", [["--threads", "0"], ["--runs", "0"], ["--ci", "--device", "cuda"], ["--device", "gpu"]]
)
def test_main_refuses_bad_options_before_loading_anything(extra: list[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as e:  # the package directory does not exist: argument checks come first
        q.main([str(tmp_path / "no-voice"), str(CORPUS), *extra])
    assert e.value.code == 2


def test_failures_refuse_a_language_without_a_bar() -> None:
    with pytest.raises(KeyError):
        q.failures(rtf=0.1, wer=0.0, rtf_key="cpu", language="xx", thresholds=_thresholds())
