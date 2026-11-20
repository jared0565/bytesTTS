# SPDX-License-Identifier: MPL-2.0
"""tools/convert/voice_details.py builds details.json from the committed VCTK table, then pins the sample clips."""

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from bytestts.catalog.catalog import load_catalog
from bytestts.catalog.details import parse_details
from bytestts.catalog.manifest import FileEntry
from bytestts.errors import IntegrityError

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "tools" / "convert" / "data" / "vctk-speaker-info.txt"
HF = "https://huggingface.co/fbmacagba"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("voice_details", ROOT / "tools" / "convert" / "voice_details.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vd = _load()

HEADER = "ID  AGE  GENDER  ACCENTS  REGION COMMENTS \n"
ROWS = (
    "p225  23  F    English    Southern  England\n"  # a two-word region
    "p226  22  M    English    Surrey\n"  # plain
    "p248  23  F    Indian  \n"  # no region
    "p261  26  F    NorthernIrish  Belfast\n"  # a CamelCase accent
    "p280  25  F    Unknown  France (mic2 files unavailable)  \n"  # Unknown, with a comment
    "p315  18  M    American  New  England  (Text and mic2 files unavailable) \n"
    "p901  30  M    SouthAfrican  Cape  Town\n"
    "p902  30  F    NewZealand  \n"
    "s5    22  F    British (new speaker, more data would be released for training a speaker-dependent model) \n"
)


def test_parse_speaker_info_shapes() -> None:
    rows = vd.parse_speaker_info(HEADER + ROWS)
    assert rows["p225"] == {"gender": "Female", "age": 23, "accent": "English (England)", "region": "Southern England"}
    assert rows["p226"] == {"gender": "Male", "age": 22, "accent": "English (England)", "region": "Surrey"}
    assert rows["p248"] == {"gender": "Female", "age": 23, "accent": "Indian"}
    assert rows["p261"] == {"gender": "Female", "age": 26, "accent": "Northern Irish", "region": "Belfast"}
    assert rows["p280"] == {"gender": "Female", "age": 25, "region": "France"}
    assert rows["p315"] == {"gender": "Male", "age": 18, "accent": "American", "region": "New England"}
    assert rows["p901"]["accent"] == "South African" and rows["p901"]["region"] == "Cape Town"
    assert rows["p902"] == {"gender": "Female", "age": 30, "accent": "New Zealand"}
    assert rows["s5"] == {"gender": "Female", "age": 22, "accent": "British"}
    assert list(rows) == ["p225", "p226", "p248", "p261", "p280", "p315", "p901", "p902", "s5"]


@pytest.mark.parametrize(
    "row",
    ["p1  x  F    English\n", "p1  22  X    English\n", "p1  22  F\n", "p225  23  F    English\n"],
    ids=["age", "gender", "no-accent", "duplicate"],
)
def test_parse_refuses_a_bad_row(row: str) -> None:
    with pytest.raises(IntegrityError):
        vd.parse_speaker_info(HEADER + ROWS + row)


def test_read_refuses_wrong_hash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "speaker-info.txt"
    bad.write_bytes(TABLE.read_bytes().replace(b"\n", b"\r\n"))  # what a CRLF checkout would make of it
    with pytest.raises(IntegrityError, match="sha256"):
        vd.read_speaker_info(bad)
    out = tmp_path / "details.json"
    assert vd.main([str(out), "--speaker-info", str(bad)]) != 0
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "sha256" in err
    assert not out.exists()


def test_committed_table_matches_its_pin() -> None:  # fails if a checkout turned its LF endings into CRLF
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == vd.SPEAKER_INFO_SHA256
    assert vd.read_speaker_info(TABLE).startswith("ID  AGE  GENDER")


def _catalogue(vctk_speakers: list[str]) -> dict[str, Any]:
    catalogue = load_catalog()
    featured = [s for s in catalogue["en-vctk"].featured if s in vctk_speakers]
    catalogue["en-vctk"] = dataclasses.replace(catalogue["en-vctk"], speakers=vctk_speakers, featured=featured)
    return catalogue


def test_build_marks_unlisted_speakers() -> None:
    rows = vd.parse_speaker_info(HEADER + ROWS)
    rows = {k: v for k, v in rows.items() if k in ("p225", "p280", "p315", "s5")}
    out = vd.build(rows, _catalogue(["ED", "p280", "p225"]))
    speakers = out["voices"]["en-vctk"]["speakers"]
    assert list(speakers) == ["ED", "p280", "p225"]  # the model's order
    assert speakers["ED"] == {"note": "not in the VCTK speaker table"}  # in the model, not in the table
    assert speakers["p225"] == rows["p225"] and speakers["p280"] == rows["p280"]  # p315 and s5: in the table only
    assert list(out["voices"]) == list(load_catalog())  # catalogue order
    parse_details(json.dumps(out).encode())
    with pytest.raises(IntegrityError, match="p999"):  # another row the model lacks
        vd.build({**rows, "p999": rows["p225"]}, _catalogue(["ED", "p280", "p225"]))
    with pytest.raises(IntegrityError, match="p226"):  # another model speaker the table lacks
        vd.build(rows, _catalogue(["ED", "p280", "p225", "p226"]))


def _write_listings(samples: Path, listings: dict[str, list[dict[str, Any]]]) -> None:
    for voice_id, rows in listings.items():
        (samples / voice_id).mkdir(parents=True)
        (samples / voice_id / "listing.json").write_text(json.dumps(rows), encoding="utf-8")


LISTINGS = {
    "en-vctk": [{"voice": "en-vctk", "name": "p228", "file": "samples/p228.wav", "size": 140000, "sha256": "b" * 64}],
    "en-ljspeech": [
        {"voice": "en-ljspeech", "name": "reader", "file": "samples/reader.wav", "size": 120000, "sha256": "c" * 64}
    ],
}


def test_pin_adds_samples(tmp_path: Path) -> None:
    out = tmp_path / "details.json"
    assert vd.main([str(out)]) == 0  # the facts, from the committed table
    _write_listings(tmp_path / "samples", LISTINGS)
    argv = [
        "--pin",
        str(tmp_path / "samples"),
        "--commit",
        "en-vctk=" + "1" * 40,
        "--commit",
        "en-ljspeech=" + "2" * 40,
    ]
    assert vd.main([*argv, str(out)]) == 0
    d = parse_details(out.read_bytes())
    vctk, lj = d.voice("en-vctk"), d.voice("en-ljspeech")
    assert vctk is not None and lj is not None and lj.reader is not None
    p228 = vctk.person("p228")
    assert p228 is not None and p228.sample == FileEntry(
        "b" * 64, 140000, f"{HF}/bytestts-en-vctk/resolve/{'1' * 40}/samples/p228.wav"
    )
    assert lj.reader.sample == FileEntry(
        "c" * 64, 120000, f"{HF}/bytestts-en-ljspeech/resolve/{'2' * 40}/samples/reader.wav"
    )
    p225 = vctk.person("p225")
    assert p225 is not None and p225.sample is None and p225.accent == "English (England)"  # facts kept


@pytest.mark.parametrize(
    ("listings", "commits", "match"),
    [
        (LISTINGS, {"en-vctk": "1" * 40}, "en-ljspeech"),  # a listed voice without --commit
        (LISTINGS, {"en-vctk": "1" * 40, "en-ljspeech": "abc"}, "commit"),
        (LISTINGS, {"en-vctk": "1" * 40, "en-ljspeech": "A" * 40}, "commit"),
        ({"en-vctk": [{**LISTINGS["en-vctk"][0], "name": "p999"}]}, {"en-vctk": "1" * 40}, "p999"),
        ({"en-vctk": [{**LISTINGS["en-vctk"][0], "name": "reader"}]}, {"en-vctk": "1" * 40}, "reader"),
        ({"en-ljspeech": [{**LISTINGS["en-ljspeech"][0], "name": "p228"}]}, {"en-ljspeech": "2" * 40}, "p228"),
        ({"en-vctk": [{**LISTINGS["en-vctk"][0], "voice": "es-css10"}]}, {"en-vctk": "1" * 40}, "es-css10"),
        ({}, {"en-vctk": "1" * 40}, "en-vctk"),  # a --commit for a voice with no listing
    ],
    ids=["no-commit", "short-sha", "upper-sha", "unknown-speaker", "reader-on-speakers", "speaker-on-reader",
         "wrong-voice", "commit-without-listing"],
)  # fmt: skip
def test_pin_refuses(listings: dict[str, list[dict[str, Any]]], commits: dict[str, str], match: str) -> None:
    details = vd.build(vd.parse_speaker_info(vd.read_speaker_info(TABLE)), load_catalog())
    with pytest.raises(IntegrityError, match=match):
        vd.pin(details, listings, commits)
