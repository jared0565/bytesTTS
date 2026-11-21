# SPDX-License-Identifier: MPL-2.0
"""Makes and checks the voice sample clips the panel's ▶ Sample and ▶ Preview play (voice labels design §6.5). A
developer tool, not shipped.

  pwsh -NoProfile -File docs\\internal\\spike\\pinned.ps1 LOG uv run --no-sync python -P tools/dev/make_samples.py OUT
      [--voice ID]

A real-voice run: only with the operator's OK, and the routine first (backup, spin check, load_monitor.py,
top_counters.ps1), pinned, one voice at a time. The voices must already be downloaded: nothing is fetched here.

For each voice it loads the voice once (CPU, 1 thread) and speaks the language's sentence at speed 1.0 for every
speaker, or once as `reader` for a voice without a speaker table, through the same stream and join the panel's
Speak uses (bytestts/gui/jobs.py), with no normalising or trimming: a clip is exactly what the user would get. It
writes OUT/<id>/samples/<name>.wav and checks every clip: 16-bit mono at the voice's rate, 1 to 8 s, RMS of at least
1% of full scale (-40 dBFS), and under 0.1% of samples at full scale. It names every failure and then writes no
OUT/<id>/listing.json for that voice. The listing (voice, name, file, size, sha256 per clip) is what
tools/convert/voice_details.py --pin reads. These limits are starting values: a real clip that trips one goes to the
operator, never a quietly loosened limit.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import wave
from pathlib import Path

from bytestts.api import BytesTTS
from bytestts.audio import concat
from bytestts.catalog import catalog, details
from bytestts.catalog.manifest import safe_name
from bytestts.gui import jobs

MIN_SECONDS = 1.0
MAX_SECONDS = 8.0
MIN_RMS = 0.01  # of full scale: -40 dBFS
MAX_CLIPPED = 0.001  # the share of samples at full scale
READER = "reader"  # the clip name of a voice without a speaker table


def check_clip(wav: bytes, rate: int) -> list[str]:
    """What is wrong with a clip, one line each; empty when it is fine. `rate` is the voice's sample rate."""
    try:
        with wave.open(io.BytesIO(wav), "rb") as w:
            channels, width, framerate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError, TypeError, ValueError) as e:
        return [f"not a WAV file: {e}"]
    if channels != 1:
        return [f"{channels} channels: a clip must be mono"]
    if width != 2:
        return [f"{8 * width}-bit: a clip must be 16-bit"]
    problems = []
    if framerate != rate:
        problems.append(f"{framerate} Hz: the voice's rate is {rate} Hz")
    samples = array.array("h", frames[: len(frames) - len(frames) % 2])
    if sys.byteorder == "big":
        samples.byteswap()  # WAV is little-endian
    seconds = len(samples) / framerate if framerate > 0 else 0.0
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        problems.append(f"{seconds:.2f} s long: a clip must be 1 to 8 s")
    n = max(len(samples), 1)
    rms = math.sqrt(sum(s * s for s in samples) / n) / 32768
    clipped = sum(1 for s in samples if s >= 32767 or s <= -32767) / n
    if rms < MIN_RMS:
        problems.append(f"RMS {100 * rms:.2f}% of full scale: under the 1% (-40 dBFS) minimum, so near silent")
    if clipped >= MAX_CLIPPED:
        problems.append(f"{100 * clipped:.2f}% of samples at full scale: 0.1% or more means clipped")
    return problems


def clip_for(tts: BytesTTS, sentence: str, speaker: str | None) -> bytes:
    """The WAV the panel's Speak would make of `sentence` at speed 1.0: the same stream, joined the same way."""
    chunks = list(tts.stream(sentence, speaker=speaker, speed=1.0))
    return concat(chunks, pause_s=BytesTTS.SENTENCE_PAUSE_S).to_wav_bytes()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the folder for <voice-id>/samples/ and <voice-id>/listing.json")
    ap.add_argument("--voice", action="append", metavar="ID", help="only this voice (repeat for more)")
    args = ap.parse_args(argv)
    voices = catalog.load_catalog()
    found = details.load_details()
    wanted = args.voice or list(voices)
    if unknown := [voice_id for voice_id in wanted if voice_id not in voices]:
        print(f"make_samples: unknown voice {', '.join(unknown)}", file=sys.stderr)
        return 1
    failed = False
    for voice_id in wanted:
        manifest = voices[voice_id]
        sentence = found.sentence(manifest.language)
        if sentence is None:
            print(f"make_samples: {voice_id}: no sentence for language {manifest.language}", file=sys.stderr)
            failed = True
            continue
        folder = args.out / voice_id
        (folder / "listing.json").unlink(missing_ok=True)  # a run that fails must leave no listing behind
        (folder / "samples").mkdir(parents=True, exist_ok=True)
        for old in (folder / "samples").glob("*.wav"):
            old.unlink()  # a clip from an earlier run must not be uploaded with these
        tts = jobs.load_voice(voice_id, "cpu", 1)
        rows, problems = [], []
        for name in list(manifest.speakers) or [READER]:
            wav = clip_for(tts, sentence, name if manifest.speakers else None)
            file = f"samples/{safe_name(name)}.wav"
            (folder / file).write_bytes(wav)
            problems += [f"{voice_id}/{name}: {p}" for p in check_clip(wav, manifest.sample_rate)]
            sha = hashlib.sha256(wav).hexdigest()
            rows.append({"voice": voice_id, "name": name, "file": file, "size": len(wav), "sha256": sha})
        del tts
        for problem in problems:
            print(f"make_samples: {problem}", file=sys.stderr)
        if problems:
            failed = True
            continue
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".listing-", suffix=".json")
        with os.fdopen(fd, "wb") as f:
            f.write((json.dumps(rows, indent=2, ensure_ascii=True) + "\n").encode("ascii"))
        os.replace(tmp, folder / "listing.json")
        print(f"{voice_id}\t{len(rows)} clips", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
