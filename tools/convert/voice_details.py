# SPDX-License-Identifier: MPL-2.0
"""Write src/bytestts/catalog/details.json: the voice and speaker facts, then (second mode) the sample pins.

  uv run --no-sync python -P tools/convert/voice_details.py OUT [--speaker-info PATH]
  uv run --no-sync python -P tools/convert/voice_details.py --pin SAMPLES_DIR --commit ID=SHA [--commit ...] OUT

The first mode writes the facts, with no samples: en-vctk's speakers from the committed VCTK table
(tools/convert/data/vctk-speaker-info.txt, refused unless its sha256 is the pin), and the other voices' names and
readers from the sourced constants below. The second mode adds the samples to an existing OUT from every
SAMPLES_DIR/<voice-id>/listing.json that tools/dev/make_samples.py wrote, with one --commit per voice repo (the
upload's 40-hex commit). Either way, the package's own parser checks the result before it is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bytestts.catalog.catalog import load_catalog
from bytestts.catalog.details import parse_details
from bytestts.catalog.manifest import VoiceManifest
from bytestts.errors import IntegrityError

ROOT = Path(__file__).resolve().parents[2]
SPEAKER_INFO = ROOT / "tools" / "convert" / "data" / "vctk-speaker-info.txt"
# speaker-info.txt, the member of VCTK-Corpus-0.92.zip on Edinburgh DataShare (4,028 bytes), read 2026-11-20
SPEAKER_INFO_URL = "https://datashare.ed.ac.uk/handle/10283/3443"
SPEAKER_INFO_SHA256 = "ee6447edab776828e0c2e726c04a4965c4602c32047b14fac2f5294d18728eee"
VCTK = "en-vctk"
NOT_IN_TABLE = "not in the VCTK speaker table"  # nothing published says who the model's row 0 ("ED") is
TABLE_ONLY = frozenset(("p315", "s5"))  # rows of the table the model has no speaker for
MODEL_ONLY = frozenset(("ED",))  # the model's speaker the table does not describe

# Each voice's language_name and dataset, as its label shows them.
VOICES = {
    # https://keithito.com/LJ-Speech-Dataset/ does not say the reader speaks American English, so no "(US)".
    "en-ljspeech": ("English", "LJSpeech"),
    # Not "English (UK)": 39 of the 108 described speakers have accents from outside the UK and Ireland.
    "en-vctk": ("English", "VCTK"),
    "es-css10": ("Spanish", "CSS10"),
    "fr-css10": ("French", "CSS10"),
}
# What a published source states about a voice's one reader (or the one speaker of a one-speaker table).
PEOPLE: dict[str, dict[str, Any]] = {
    # The CSS10 paper (Park and Mulc, 2019, https://arxiv.org/abs/1903.11269, section 2): the LJ Speech dataset
    # "consists of audio files segmented from audiobooks in LibriVox recorded by a female volunteer".
    "en-ljspeech": {"gender": "Female"},
    # es-css10 (tux) and fr-css10 (gilles): https://github.com/Kyubyong/css10 names the readers (Tux; Gilles G. Le
    # Blanc), and neither it, the CSS10 paper nor their LibriVox reader pages (https://librivox.org/reader/3946 and
    # https://librivox.org/reader/7423) state anything more about them.
}
# The sample sentence per language, the same for the clips and the live preview (approved by the operator, 2026-11-20).
SENTENCES = {
    "en": "Hello. This is how I sound reading your text aloud.",
    "es": "Hola. Así sueno al leer tu texto en voz alta.",
    "fr": "Bonjour. Voici ma voix quand je lis votre texte à haute voix.",
}
GENDERS = {"F": "Female", "M": "Male"}
# The table's accent words, spelled out; "English" names the country, so it doesn't read as the language.
ACCENTS = {
    "English": "English (England)",
    "NorthernIrish": "Northern Irish",
    "SouthAfrican": "South African",
    "NewZealand": "New Zealand",
}
UNKNOWN = "Unknown"  # p280: left out, its region (France) is kept
HF_RESOLVE = "https://huggingface.co/fbmacagba/bytestts-{voice}/resolve/{commit}/samples/{name}.wav"
_COMMENT = re.compile(r"\s*\([^)]*\)\s*$")  # a trailing "(mic2 files unavailable)"
_COMMIT = re.compile(r"[0-9a-f]{40}")


def read_speaker_info(path: Path) -> str:
    """The VCTK table's text; IntegrityError unless its bytes are the pinned ones."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SPEAKER_INFO_SHA256:
        raise IntegrityError(
            f"{path}: its sha256 is {digest}, not the pinned {SPEAKER_INFO_SHA256} "
            f"(speaker-info.txt from {SPEAKER_INFO_URL}, unmodified, with LF line endings)"
        )
    return data.decode("ascii")


def parse_speaker_info(text: str) -> dict[str, dict[str, Any]]:
    """Rows of `ID AGE GENDER ACCENTS [REGION...] [(comment)]`, keyed by ID, in table order."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("ID "):
        raise IntegrityError("speaker-info: the first line is not the ID AGE GENDER ACCENTS REGION header")
    rows: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4 or not fields[1].isdigit() or fields[2] not in GENDERS:
            raise IntegrityError(f"speaker-info line {number}: not ID AGE F|M ACCENT [REGION]: {line.strip()!r}")
        name, age, gender, accent, *rest = fields
        if name in rows:
            raise IntegrityError(f"speaker-info line {number}: {name} is listed twice")
        row: dict[str, Any] = {"gender": GENDERS[gender], "age": int(age)}
        if accent != UNKNOWN:
            row["accent"] = ACCENTS.get(accent, accent)
        if region := _COMMENT.sub("", " ".join(rest)).strip():
            row["region"] = region
        rows[name] = row
    return rows


def build(rows: Mapping[str, Mapping[str, Any]], catalogue: Mapping[str, VoiceManifest]) -> dict[str, Any]:
    """The details.json document, without samples: voices in catalogue order, speakers in each model's order."""
    if missing := set(catalogue) - set(VOICES):
        raise IntegrityError(f"no language_name and dataset for {sorted(missing)}: add them to VOICES")
    voices: dict[str, Any] = {}
    for voice_id, m in catalogue.items():
        language_name, dataset = VOICES[voice_id]
        entry: dict[str, Any] = {"language_name": language_name, "dataset": dataset}
        if voice_id == VCTK:
            if unlisted := {s for s in m.speakers if s not in rows} - MODEL_ONLY:
                raise IntegrityError(f"{VCTK} speakers {sorted(unlisted)} are not in the VCTK table")
            if extra := set(rows) - set(m.speakers) - TABLE_ONLY:
                raise IntegrityError(f"VCTK table rows {sorted(extra)} are not speakers of {VCTK}")
            entry["speakers"] = {s: dict(rows[s]) if s in rows else {"note": NOT_IN_TABLE} for s in m.speakers}
        elif len(m.speakers) > 1:
            raise IntegrityError(f"{voice_id} has {len(m.speakers)} speakers and no table that describes them")
        elif m.speakers:
            entry["speakers"] = {m.speakers[0]: dict(PEOPLE.get(voice_id, {}))}
        else:
            entry["reader"] = dict(PEOPLE.get(voice_id, {}))
        voices[voice_id] = entry
    return {"sentences": dict(SENTENCES), "voices": voices}


def pin(
    details: dict[str, Any], listings: Mapping[str, list[dict[str, Any]]], commits: Mapping[str, str]
) -> dict[str, Any]:
    """`details` with a sample for every clip in `listings` (by voice id), pinned at that voice repo's commit."""
    out = copy.deepcopy(details)
    if unlisted := set(commits) - set(listings):
        raise IntegrityError(f"--commit for {sorted(unlisted)}, which has no listing.json")
    for voice_id, clips in listings.items():
        voice = out["voices"].get(voice_id)
        commit = commits.get(voice_id)
        if voice is None:
            raise IntegrityError(f"{voice_id}/listing.json: {voice_id} is not a voice in the details")
        if commit is None:
            raise IntegrityError(f"{voice_id}/listing.json: no --commit {voice_id}=<40 hex> for its clips")
        if not _COMMIT.fullmatch(commit):
            raise IntegrityError(f"--commit {voice_id}={commit}: the commit must be 40 lowercase hex digits")
        for clip in clips:
            name = clip.get("name")
            if clip.get("voice") != voice_id:
                raise IntegrityError(f"{voice_id}/listing.json lists a clip of {clip.get('voice')!r}")
            if "speakers" in voice:
                if name not in voice["speakers"]:
                    raise IntegrityError(f"{voice_id}/listing.json: {name!r} is not a speaker of {voice_id}")
                person = voice["speakers"][name]
            elif name == "reader":
                person = voice["reader"]
            else:
                raise IntegrityError(
                    f"{voice_id}/listing.json: {name!r} is not 'reader', and {voice_id} has no speakers"
                )
            if clip.get("file") != f"samples/{name}.wav":
                raise IntegrityError(f"{voice_id}/listing.json: {name!r} is filed as {clip.get('file')!r}")
            url = HF_RESOLVE.format(voice=voice_id, commit=commit, name=name)
            person["sample"] = {"sha256": clip.get("sha256"), "size": clip.get("size"), "url": url}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="details.json to write (with --pin: to add the samples to)")
    ap.add_argument(
        "--speaker-info", type=Path, default=SPEAKER_INFO, help="the VCTK table (default: the committed one)"
    )
    ap.add_argument("--pin", type=Path, metavar="SAMPLES_DIR", help="add the samples listed under SAMPLES_DIR/<id>/")
    ap.add_argument("--commit", action="append", default=[], metavar="ID=SHA", help="a voice repo's upload commit")
    args = ap.parse_args(argv)
    if args.commit and args.pin is None:
        ap.error("--commit goes with --pin")
    try:
        if args.pin is None:
            details = build(parse_speaker_info(read_speaker_info(args.speaker_info)), load_catalog())
        else:
            commits = dict(c.split("=", 1) for c in args.commit if "=" in c)
            if len(commits) != len(args.commit):
                ap.error("--commit: give each voice once, as ID=SHA")
            listings = {
                d.name: json.loads((d / "listing.json").read_text(encoding="utf-8"))
                for d in sorted(args.pin.iterdir())
                if (d / "listing.json").is_file()
            }
            details = pin(json.loads(args.out.read_text(encoding="utf-8")), listings, commits)
        text = json.dumps(details, ensure_ascii=False, indent=2) + "\n"
        parse_details(text.encode("utf-8"))  # the package's own parser has the last word
        fd, part = tempfile.mkstemp(dir=args.out.parent, prefix=args.out.name, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:  # bytes, so no CRLF on Windows
                f.write(text.encode("utf-8"))
            os.replace(part, args.out)
        except BaseException:
            Path(part).unlink(missing_ok=True)
            raise
    except (IntegrityError, OSError, ValueError, AttributeError, TypeError) as e:
        print(f"voice_details: {e}", file=sys.stderr)
        return 1
    voices = details["voices"]
    samples = sum(
        "sample" in p
        for v in voices.values()
        for p in [*v.get("speakers", {}).values(), *([v["reader"]] if "reader" in v else [])]
    )
    print(f"{args.out}: {len(voices)} voices, {samples} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
