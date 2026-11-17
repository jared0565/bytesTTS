# SPDX-License-Identifier: MPL-2.0
"""The voice catalogue that ships inside the package (voices.json)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources

from bytestts.catalog.manifest import VoiceManifest, parse_json
from bytestts.errors import IntegrityError

# Design 6.6: files are fetched from a commit, never a branch: https, /resolve/<40-hex sha>/, no query or fragment
_PINNED_URL = re.compile(r"https://[^/?#\s]+/[^?#\s]*/resolve/[0-9a-f]{40}/[^?#\s]+")


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    language: str
    speakers: list[str]
    featured: list[str]
    licence: str
    data_licence: str
    attribution: str
    sample_rate: int
    size_bytes: int

    @classmethod
    def from_manifest(cls, m: VoiceManifest) -> VoiceInfo:
        return cls(
            id=m.id,
            language=m.language,
            speakers=list(m.speakers),
            featured=list(m.featured),
            licence=m.licence,
            data_licence=m.data_licence,
            attribution=m.attribution,
            sample_rate=m.sample_rate,
            size_bytes=sum(entry.size for entry in m.files.values()),
        )


def parse_catalog(data: str | bytes) -> dict[str, VoiceManifest]:
    raw = parse_json(data.encode("utf-8") if isinstance(data, str) else data, "voices.json")
    if not isinstance(raw, dict) or set(raw) != {"voices"} or not isinstance(raw["voices"], list):
        raise IntegrityError('voices.json must be {"voices": [...]}')
    voices: dict[str, VoiceManifest] = {}
    for item in raw["voices"]:
        m = VoiceManifest.from_dict(item)
        if m.id in voices:
            raise IntegrityError(f"voices.json lists {m.id!r} twice")
        if m.origin != "catalog":
            raise IntegrityError(f"voice {m.id!r}: the catalogue ships only origin 'catalog'")
        for name, entry in m.files.items():
            if entry.url is None or not _PINNED_URL.fullmatch(entry.url):
                raise IntegrityError(
                    f"voice {m.id!r}: {name} needs an https URL pinned to a commit (/resolve/<40-hex sha>/)"
                )
        voices[m.id] = m
    return voices


def load_catalog() -> dict[str, VoiceManifest]:
    return parse_catalog(resources.files("bytestts.catalog").joinpath("voices.json").read_bytes())


def _language_matches(voice: str, wanted: str) -> bool:
    return voice == wanted or voice.startswith(wanted + "-")  # "en" takes en-gb, not eng


def list_voices(language: str | None = None) -> list[VoiceInfo]:
    """Catalogue voices, optionally for one language: those with featured speakers first, then by id."""
    wanted = language.lower() if language is not None else None
    voices = [m for m in load_catalog().values() if wanted is None or _language_matches(m.language.lower(), wanted)]
    voices.sort(key=lambda m: (not m.featured, m.id))
    return [VoiceInfo.from_manifest(m) for m in voices]
