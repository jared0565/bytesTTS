# SPDX-License-Identifier: MPL-2.0
"""Quality check for a voice package: real-time factor and word error rate against tests/quality/thresholds.json.

  uv run --group eval python -P tools/dev/quality.py PKG tests/corpus/en.txt [--device cpu|cuda] [--threads N] [--ci]

The rules were approved before any measurement:
- RTF: wall-clock synthesis time over audio duration for the whole corpus, one sentence per call. The voice loads
  untimed, one warm-up sentence runs uncounted, and the median of the runs' own ratios is reported (3 runs).
- WER: corpus level, total word edits over total reference words. Each line has a written reference and a spoken
  one (`<lang>.spoken.txt` beside the corpus) and is scored against whichever needs fewer edits, the written one on
  a tie. Both sides are normalised alike: NFC, casefold, every Unicode punctuation character to a space, whitespace
  collapsed.
- ASR: faster-whisper with the model and revision pinned in thresholds.json, int8 on CPU, beam 5, the language
  forced to the corpus language (the corpus file's stem).

--threads (default 1) applies to ONNX Runtime and to faster-whisper's cpu_threads. Prints a JSON report and exits
1 when a threshold is missed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
import time
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from bytestts.audio import Audio

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = ROOT / "tests" / "quality" / "thresholds.json"
ASR_RATE = 16000


class Run(NamedTuple):
    seconds: float  # wall-clock synthesis time for the whole corpus
    audio: list[Audio]  # one per corpus line


def normalise(text: str) -> list[str]:
    folded = unicodedata.normalize("NFC", text).casefold()
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in folded).split()


def word_edits(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Levenshtein distance over words: substitutions, insertions and deletions each cost 1."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def corpus_wer(references: Sequence[tuple[str, str]], hypotheses: Sequence[str]) -> float:
    """Total edits over total words, each line against its (written, spoken) reference with fewer edits."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(hypotheses)} hypotheses for {len(references)} corpus lines")
    edits = words = 0
    for (written, spoken), hypothesis in zip(references, hypotheses, strict=True):
        hyp = normalise(hypothesis)
        scored = [(word_edits(ref, hyp), len(ref)) for ref in (normalise(written), normalise(spoken))]
        line_edits, line_words = min(scored, key=lambda s: s[0])  # min keeps the first, the written one, on a tie
        edits += line_edits
        words += line_words
    if words == 0:
        raise ValueError("the references have no words")
    return edits / words


def corpus_language(path: Path) -> str:
    return path.name.split(".", 1)[0]


def load_corpus(path: Path) -> list[tuple[str, str]]:
    """(written, spoken) pairs from `<lang>.txt` and its `<lang>.spoken.txt` sibling, line for line."""
    spoken_path = path.with_name(f"{corpus_language(path)}.spoken.txt")
    written = path.read_text(encoding="utf-8").splitlines()
    spoken = spoken_path.read_text(encoding="utf-8").splitlines()
    if len(written) != len(spoken):
        raise ValueError(f"{path.name} has {len(written)} lines but {spoken_path.name} has {len(spoken)} lines")
    if any(not line.strip() for line in written + spoken):
        raise ValueError(f"{path.name} or {spoken_path.name} has a blank line")
    return list(zip(written, spoken, strict=True))


def measure(
    synthesize: Callable[[str], Audio],
    lines: Sequence[str],
    *,
    runs: int = 3,
    clock: Callable[[], float] = time.perf_counter,
) -> list[Run]:
    """One untimed warm-up sentence, then `runs` timed passes over every line."""
    if runs < 1 or not lines:
        raise ValueError("measure needs at least one run and one line")
    synthesize(lines[0])
    out = []
    for _ in range(runs):
        start = clock()
        audio = [synthesize(line) for line in lines]
        out.append(Run(clock() - start, audio))
    return out


def rtf(runs: Sequence[Run]) -> float:
    """The median over runs of synthesis seconds / audio seconds (a run's audio varies with the noise)."""
    if not runs:
        raise ValueError("no runs to measure")
    ratios = []
    for run in runs:
        audio_seconds = sum(a.duration for a in run.audio)
        if not audio_seconds > 0:
            raise ValueError("a run produced no audio")
        ratios.append(run.seconds / audio_seconds)
    return float(statistics.median(ratios))


def resample(samples: np.ndarray, rate: int, target: int = ASR_RATE) -> np.ndarray:
    """Band-limited resampling through the FFT (numpy only): the ASR model takes 16 kHz float32."""
    if rate == target:
        return samples
    n = round(len(samples) * target / rate)
    spectrum = np.fft.rfft(samples.astype(np.float64))
    keep = n // 2 + 1
    spectrum = spectrum[:keep] if keep <= len(spectrum) else np.pad(spectrum, (0, keep - len(spectrum)))
    return (np.fft.irfft(spectrum, n) * (n / len(samples))).astype(np.float32)


def failures(
    *, rtf: float, wer: float, rtf_key: str, language: str, thresholds: dict[str, dict[str, float]]
) -> list[str]:
    """Each missed bar, RTF first. The bars are inclusive."""
    rtf_bar, wer_bar = thresholds["rtf"][rtf_key], thresholds["wer"][language]
    found = []
    if not rtf <= rtf_bar:
        found.append(f"RTF {rtf:.3f} is above the {rtf_key} bar {rtf_bar}")
    if not wer <= wer_bar:
        found.append(f"WER {wer:.4f} is above the {language} bar {wer_bar}")
    return found


def transcribe(audio: Sequence[Audio], *, language: str, asr: dict[str, Any], threads: int) -> list[str]:
    """faster-whisper at the pinned model revision, on CPU (eval group, maintainers only)."""
    package, _, version = asr["package"].partition("==")
    installed = importlib.metadata.version(package)
    if installed != version:
        raise RuntimeError(f"{package} {installed} is installed; the thresholds were set with {version}")
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(asr["model"], revision=asr["revision"])
    model = WhisperModel(model_dir, device="cpu", compute_type=asr["compute_type"], cpu_threads=threads, num_workers=1)
    out = []
    for a in audio:
        segments, _ = model.transcribe(
            resample(a.samples, a.sample_rate), language=language, beam_size=asr["beam_size"]
        )
        out.append(" ".join(s.text.strip() for s in segments))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path, help="a voice package directory")
    ap.add_argument("corpus", type=Path, help="a corpus such as tests/corpus/en.txt")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime and ASR CPU threads (default 1)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--ci", action="store_true", help="judge RTF against the CI runner's CPU bar")
    args = ap.parse_args(argv)
    if args.threads < 1 or args.runs < 1:
        ap.error("--threads and --runs must be at least 1")
    if args.ci and args.device != "cpu":
        ap.error("--ci judges CPU synthesis only")

    from bytestts import BytesTTS

    thresholds = json.loads(THRESHOLDS.read_bytes())
    pairs = load_corpus(args.corpus)
    language = corpus_language(args.corpus)
    rtf_key = "ci-cpu" if args.ci else args.device
    tts = BytesTTS.load_path(args.package, device=args.device, threads=args.threads)
    runs = measure(tts.synthesize, [written for written, _ in pairs], runs=args.runs)
    hypotheses = transcribe(runs[-1].audio, language=language, asr=thresholds["asr"], threads=args.threads)
    value_rtf, value_wer = rtf(runs), corpus_wer(pairs, hypotheses)
    found = failures(rtf=value_rtf, wer=value_wer, rtf_key=rtf_key, language=language, thresholds=thresholds)
    report = {
        "voice": tts.info.id,
        "language": language,
        "device": args.device,
        "threads": args.threads,
        "rtf": value_rtf,
        "rtf_bar": thresholds["rtf"][rtf_key],
        "run_seconds": [r.seconds for r in runs],
        "audio_seconds": [sum(a.duration for a in r.audio) for r in runs],
        "wer": value_wer,
        "wer_bar": thresholds["wer"][language],
        "transcripts": hypotheses,
        "failures": found,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
