# SPDX-License-Identifier: MPL-2.0
"""Voice and speaker details: the labels, the filter facts and the sample clips the control panel shows.

The packaged `details.json` is joined to the catalogue by voice id when it is used. Nothing in it reaches a
`VoiceManifest` or the cache key, so it can change without any installed voice downloading again. A catalogue voice
with no entry falls back to its id, with no sample and no filters.

The module's top level holds only what callers use (the background drain fuzzes top-level functions): the parse
helpers live inside `parse_details`, and every parse failure is `IntegrityError`.
"""

from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any, NoReturn

from bytestts.catalog.cache import cache_root
from bytestts.catalog.catalog import _PINNED_URL
from bytestts.catalog.download import download_file
from bytestts.catalog.manifest import FileEntry, parse_json, safe_name
from bytestts.errors import IntegrityError

SEPARATOR = " · "
MAX_DETAILS_BYTES = 1 << 20
MAX_SAMPLE_BYTES = 1 << 20
MAX_TEXT = 80  # accent, region, note, language_name, dataset
MAX_SENTENCE = 200
_GENDERS = ("Female", "Male")
_AGES = range(1, 121)
_LANGUAGE_KEY = re.compile(r"[a-z]{2,3}")  # a primary language subtag: "en", never "en-gb"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BANNED = frozenset(("Cc", "Zl", "Zp"))  # control characters and line breaks: the CLI prints tab-separated columns
_PERSON_KEYS = frozenset(("gender", "age", "accent", "region", "note", "sample"))
_SAMPLE_KEYS = frozenset(("sha256", "size", "url"))
_VOICE_KEYS = frozenset(("language_name", "dataset", "speakers", "reader"))
_TOP_KEYS = frozenset(("sentences", "voices"))

# Read when sample_bytes runs, not bound as defaults, so tests can shorten them.
SAMPLE_TIMEOUT: tuple[float, float] = (10.0, 30.0)  # connect, read
LOCK_WAIT_S: float = 30.0  # how long a second request for the same clip waits for the first
_clip_locks: dict[str, threading.Lock] = {}  # one per clip (by sha256), so a stalled clip holds up only itself
_clip_locks_guard = threading.Lock()


@dataclass(frozen=True)
class Person:
    """What a published source states about one speaker, or about the one reader of a voice without speakers."""

    gender: str | None = None
    age: int | None = None
    accent: str | None = None
    region: str | None = None
    note: str | None = None
    sample: FileEntry | None = None

    def describe(self) -> str:
        """The label without the name: `Female, 22 · English (England), Southern England`, or "" if nothing is known."""
        who = ", ".join(part for part in (self.gender, None if self.age is None else str(self.age)) if part)
        where = ", ".join(part for part in (self.accent, self.region) if part)
        return SEPARATOR.join(part for part in (who, where, self.note) if part)

    def label(self, name: str) -> str:
        described = self.describe()
        return f"{name}{SEPARATOR}{described}" if described else name


@dataclass(frozen=True)
class VoiceDetails:
    """One voice's details: `speakers` when its catalogue entry has a speaker table, else its one `reader`."""

    language_name: str
    dataset: str
    speakers: Mapping[str, Person] | None
    reader: Person | None

    def __post_init__(self) -> None:
        if (self.speakers is None) == (self.reader is None):
            raise IntegrityError(f"{self.dataset}: a voice has either speakers or a reader, not both or neither")

    def reader_label(self) -> str | None:
        return None if self.reader is None else self.reader.label(self.dataset)

    def person(self, name: str) -> Person | None:
        return None if self.speakers is None else self.speakers.get(name)


@dataclass(frozen=True)
class Details:
    sentences: Mapping[str, str]  # by primary language subtag
    voices: Mapping[str, VoiceDetails]  # by voice id

    def sentence(self, language: str) -> str | None:
        """The sample sentence for a language tag, by its primary subtag: `en-gb` gets the `en` sentence."""
        return self.sentences.get(language.split("-", 1)[0].lower())

    def voice(self, voice_id: str) -> VoiceDetails | None:
        return self.voices.get(voice_id)


def parse_details(data: bytes) -> Details:
    """`details.json`, checked strictly: UTF-8, at most 1 MiB, no duplicate or unknown keys, the right types."""

    def fail(what: str, why: str) -> NoReturn:
        raise IntegrityError(f"details.json: {what} {why}")

    def obj(value: Any, what: str, allowed: frozenset[str], required: frozenset[str] = frozenset()) -> dict[str, Any]:
        if not isinstance(value, dict):
            fail(what, "must be a JSON object")
        if missing := required - value.keys():
            fail(what, f"lacks {sorted(missing)}")
        if unknown := value.keys() - allowed:
            fail(what, f"has unknown keys {sorted(unknown)}")
        return value

    def text(value: Any, what: str, limit: int) -> str:
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= limit
            or any(unicodedata.category(c) in _BANNED for c in value)
        ):
            fail(what, f"must be 1-{limit} characters, with no control character or line break")
        return value

    def sample(value: Any, what: str) -> FileEntry:
        s = obj(value, what, _SAMPLE_KEYS, _SAMPLE_KEYS)
        sha256, size, url = s["sha256"], s["size"], s["url"]
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            fail(what, "sha256 must be 64 lowercase hex digits")
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_SAMPLE_BYTES:
            fail(what, f"size must be from 1 to {MAX_SAMPLE_BYTES} bytes")
        if not isinstance(url, str) or not _PINNED_URL.fullmatch(url):
            fail(what, "url must be an https link pinned to a commit (/resolve/<40 hex>/)")
        folder, file = url.rsplit("/", 2)[1:]
        if folder != "samples" or not file.endswith(".wav"):
            fail(what, "url must end in /samples/<name>.wav")
        try:
            safe_name(file.removesuffix(".wav"))
        except IntegrityError as e:
            fail(what, f"url names an unsafe file: {e}")
        return FileEntry(sha256, size, url)

    def person(value: Any, what: str) -> Person:
        p = obj(value, what, _PERSON_KEYS)
        if "gender" in p and p["gender"] not in _GENDERS:
            fail(what, 'gender must be "Female" or "Male"')
        age = p.get("age")
        if "age" in p and (isinstance(age, bool) or not isinstance(age, int) or age not in _AGES):
            fail(what, "age must be a whole number from 1 to 120")
        said = {key: text(p[key], f"{what} {key}", MAX_TEXT) for key in ("accent", "region", "note") if key in p}
        return Person(
            gender=p.get("gender"),
            age=age,
            accent=said.get("accent"),
            region=said.get("region"),
            note=said.get("note"),
            sample=sample(p["sample"], f"{what} sample") if "sample" in p else None,
        )

    def voice(value: Any, what: str) -> VoiceDetails:
        v = obj(value, what, _VOICE_KEYS, frozenset(("language_name", "dataset")))
        if ("speakers" in v) == ("reader" in v):
            fail(what, "must have exactly one of speakers and reader")
        speakers = None
        if "speakers" in v:
            if not isinstance(v["speakers"], dict):
                fail(f"{what} speakers", "must be a JSON object")
            speakers = MappingProxyType(
                {name: person(p, f"{what} speaker {name!r}") for name, p in v["speakers"].items()}
            )
        return VoiceDetails(
            language_name=text(v["language_name"], f"{what} language_name", MAX_TEXT),
            dataset=text(v["dataset"], f"{what} dataset", MAX_TEXT),
            speakers=speakers,
            reader=person(v["reader"], f"{what} reader") if "reader" in v else None,
        )

    if not isinstance(data, (bytes, bytearray)):
        raise IntegrityError("details.json must be read as bytes")
    if len(data) > MAX_DETAILS_BYTES:
        raise IntegrityError(f"details.json is too large (over {MAX_DETAILS_BYTES} bytes)")
    try:
        root = obj(parse_json(bytes(data), "details.json"), "the top level", _TOP_KEYS, _TOP_KEYS)
        if not isinstance(root["sentences"], dict):
            fail("sentences", "must be a JSON object")
        sentences = {}
        for key, sentence in root["sentences"].items():
            if not _LANGUAGE_KEY.fullmatch(key):
                fail(f"sentence key {key!r}", "must be a primary language subtag such as en")
            sentences[key] = text(sentence, f"sentence {key!r}", MAX_SENTENCE)
        if not isinstance(root["voices"], dict):
            fail("voices", "must be a JSON object")
        voices = {voice_id: voice(v, f"voice {voice_id!r}") for voice_id, v in root["voices"].items()}
    except IntegrityError:
        raise
    except (TypeError, ValueError, RecursionError) as e:  # a shape no check above foresaw is still a bad file
        raise IntegrityError(f"details.json is malformed: {e}") from None
    return Details(MappingProxyType(sentences), MappingProxyType(voices))


def load_details() -> Details:
    """The packaged `details.json`, read and checked on every call, as `catalog.load_catalog` does."""
    return parse_details(resources.files("bytestts.catalog").joinpath("details.json").read_bytes())


def sample_bytes(entry: FileEntry) -> bytes:
    """A sample clip's pinned bytes, from the cache under `samples/` when it holds them, else fetched first.

    One fetch per clip at a time: a second request for the same clip waits up to LOCK_WAIT_S for the first, then
    raises TimeoutError("timed out"). The bytes returned are read once and checked against the pin, so a cached file
    changed after the check is fetched again rather than sent.
    """
    if entry.url is None or not _SHA256.fullmatch(entry.sha256):  # the sha256 names the cached file
        raise IntegrityError("a sample needs a url and a sha256 of 64 lowercase hex digits")
    with _clip_locks_guard:
        lock = _clip_locks.setdefault(entry.sha256, threading.Lock())
    if not lock.acquire(timeout=LOCK_WAIT_S):
        raise TimeoutError("timed out")
    try:
        path = cache_root() / "samples" / f"{entry.sha256}.wav"
        for _ in range(2):
            download_file(entry.url, path, sha256=entry.sha256, size=entry.size, timeout=SAMPLE_TIMEOUT)
            with path.open("rb") as f:
                data = f.read(entry.size + 1)
            if len(data) == entry.size and hashlib.sha256(data).hexdigest() == entry.sha256:
                return data
            path.unlink(missing_ok=True)  # changed after download_file checked it: fetch it once more
        raise IntegrityError(f"{path.name}: the cached sample keeps changing after it is checked")
    finally:
        lock.release()


def voice_label(voice_id: str, speakers: Sequence[str], d: VoiceDetails | None) -> str:
    """`English · VCTK · 109 speakers, 11 accents`; the voice id when it has no details.

    `speakers` is the catalogue's table, counted as `bytestts voices` counts it (an empty table is one speaker). The
    accents are counted from the details, and shown only when there is more than one.
    """
    if d is None:
        return voice_id
    count = max(1, len(speakers))
    label = SEPARATOR.join((d.language_name, d.dataset, f"{count} speaker" if count == 1 else f"{count} speakers"))
    accents = {p.accent for p in (d.speakers or {}).values() if p.accent}
    return f"{label}, {len(accents)} accents" if len(accents) > 1 else label
