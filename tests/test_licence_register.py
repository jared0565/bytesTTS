# SPDX-License-Identifier: MPL-2.0
"""The licence register (spec 9.3): every voice BytesTTS ships or is about to ship has a row in docs/licences.md
whose model licence, data licence and attribution are exactly the strings its manifest carries.

Voices come from the packaged voices.json and from pending entries in tools/convert/voices/ (a converted package's
voice.json, kept there until its files have commit-pinned URLs and it moves into the catalogue).
"""

import json
from pathlib import Path

import pytest

from bytestts.catalog.manifest import VoiceManifest

REPO = Path(__file__).resolve().parents[1]
REGISTER = REPO / "docs" / "licences.md"
CATALOGUE = REPO / "src" / "bytestts" / "catalog" / "voices.json"
PENDING = REPO / "tools" / "convert" / "voices"
COLUMNS = ["Voice", "Model licence", "Data licence", "Attribution"]


def _voices() -> dict[str, VoiceManifest]:
    voices = {}
    for item in json.loads(CATALOGUE.read_bytes())["voices"]:
        m = VoiceManifest.from_dict(item)
        voices[m.id] = m
    for path in sorted(PENDING.glob("*.voice.json")):
        m = VoiceManifest.from_dict(json.loads(path.read_bytes()))
        assert path.name == f"{m.id}.voice.json", f"{path.name} holds voice {m.id!r}"
        assert m.id not in voices, f"{m.id} is both in the catalogue and pending"
        voices[m.id] = m
    return voices


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _register() -> dict[str, list[str]]:
    lines = [ln for ln in REGISTER.read_text(encoding="utf-8").splitlines() if ln.lstrip().startswith("|")]
    assert lines and _cells(lines[0]) == COLUMNS, f"the register's table header must be {COLUMNS}"
    rows: dict[str, list[str]] = {}
    for line in lines[2:]:  # header, then the |---| separator
        cells = _cells(line)
        assert len(cells) == len(COLUMNS), f"malformed row: {line!r}"
        voice = cells[0].strip("`")
        assert voice not in rows, f"{voice} has two rows"
        rows[voice] = cells[1:]
    return rows


def test_register_covers_en_ljspeech() -> None:  # the checks below must not pass on an empty voice list
    assert "en-ljspeech" in _voices()


@pytest.mark.parametrize("voice_id", sorted(_voices()))
def test_voice_has_its_manifest_strings_in_the_register(voice_id: str) -> None:
    m = _voices()[voice_id]
    rows = _register()
    assert voice_id in rows, f"docs/licences.md has no row for {voice_id}"
    assert rows[voice_id] == [m.licence, m.data_licence, m.attribution]


@pytest.mark.parametrize("voice_id", sorted(_voices()))
def test_notice_carries_the_voice_attribution(voice_id: str) -> None:  # design: NOTICE credits every voice
    assert _voices()[voice_id].attribution in (REPO / "NOTICE").read_text(encoding="utf-8")
