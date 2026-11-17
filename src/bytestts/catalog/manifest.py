# SPDX-License-Identifier: MPL-2.0
"""Voice manifests (voice.json) and voice-package verification.

A manifest pins every file of a voice by size and sha256. Every name that becomes a path (each `files` key, and
the voice id, which Task 13 makes a cache directory) goes through safe_name first.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from bytestts.errors import ConsentMissing, IntegrityError

MANIFEST = "voice.json"
CARD = "card.json"
REQUIRED_FILES = ("model.onnx", "config.json")  # what every Phase 1 loader reads: never read unpinned
MAX_JSON_BYTES = 1 << 20
_CHUNK = 1 << 20

Origin = Literal["catalog", "fine-tuned"]
_ORIGINS = ("catalog", "fine-tuned")
_SAFE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_WINDOWS_DEVICE = re.compile(r"(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?", re.IGNORECASE)  # opens a device
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LANGUAGE = re.compile(r"[a-z]{2,3}(-[A-Za-z0-9]{1,8})*")
_KEYS = frozenset(
    (
        "id", "version", "language", "engine", "sample_rate", "licence", "data_licence", "attribution", "origin",
        "files", "speakers", "featured",
    )
)  # fmt: skip


def safe_name(name: str) -> str:
    """`name` if it is a plain file name on every OS, else IntegrityError.

    1-128 of [A-Za-z0-9._-], not `.` or `..`, no trailing dot (Windows drops it, so `a.` would open `a`), and not a
    Windows device name such as CON or nul.onnx.
    """
    if (
        not isinstance(name, str)
        or not _SAFE.fullmatch(name)
        or name in (".", "..")
        or name.endswith(".")
        or _WINDOWS_DEVICE.fullmatch(name)
    ):
        raise IntegrityError(f"unsafe file name {name!r}: use 1-128 of A-Z a-z 0-9 . _ -")
    return name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


# --- strict JSON ----------------------------------------------------------------------------------------------


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:  # json.loads would silently keep the last one
            raise IntegrityError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def parse_json(data: bytes, what: str) -> Any:
    """UTF-8 JSON with no duplicate keys."""
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise IntegrityError(f"{what} is not valid UTF-8 JSON: {e}") from None


def read_json(path: Path, what: str) -> Any:
    try:
        with Path(path).open("rb") as f:
            data = f.read(MAX_JSON_BYTES + 1)
    except OSError as e:
        raise IntegrityError(f"cannot read {what}: {e.strerror or e}") from None
    if len(data) > MAX_JSON_BYTES:
        raise IntegrityError(f"{what} is too large (over {MAX_JSON_BYTES} bytes)")
    return parse_json(data, what)


def _object(value: Any, what: str, keys: frozenset[str], optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IntegrityError(f"{what} must be a JSON object")
    if missing := keys - value.keys():
        raise IntegrityError(f"{what} lacks {sorted(missing)}")
    if unknown := value.keys() - keys - optional:
        raise IntegrityError(f"{what} has unknown keys {sorted(unknown)}")
    return value


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"{what} must be a non-empty string")
    return value


def _int(value: Any, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IntegrityError(f"{what} must be an integer >= {minimum}, not {value!r}")
    return value


def _names(value: Any, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise IntegrityError(f"{what} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise IntegrityError(f"{what} lists a name twice")
    return value


# --- the manifest ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FileEntry:
    sha256: str
    size: int
    url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise IntegrityError(f"sha256 must be 64 lowercase hex digits, not {self.sha256!r}")
        _int(self.size, "size", 0)
        if self.url is not None and not isinstance(self.url, str):
            raise IntegrityError(f"url must be a string, not {self.url!r}")

    @classmethod
    def from_dict(cls, data: Any, what: str) -> FileEntry:
        d = _object(data, what, frozenset(("sha256", "size")), frozenset(("url",)))
        return cls(d["sha256"], d["size"], d.get("url"))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sha256": self.sha256, "size": self.size}
        if self.url is not None:
            out["url"] = self.url
        return out


@dataclass(frozen=True)
class VoiceManifest:
    id: str
    version: str
    language: str
    engine: str
    sample_rate: int
    licence: str
    data_licence: str
    attribution: str
    origin: Origin
    files: Mapping[str, FileEntry]
    speakers: list[str]
    featured: list[str]

    def __post_init__(self) -> None:
        safe_name(self.id)  # a cache directory in Task 13
        for name in ("version", "engine", "licence", "data_licence", "attribution"):
            _text(getattr(self, name), name)
        if not isinstance(self.language, str) or not _LANGUAGE.fullmatch(self.language):
            raise IntegrityError(f"language must be a language tag such as en or en-gb, not {self.language!r}")
        _int(self.sample_rate, "sample_rate", 1)
        if self.origin not in _ORIGINS:
            raise IntegrityError(f"origin must be one of {_ORIGINS}, not {self.origin!r}")
        if not isinstance(self.files, Mapping):
            raise IntegrityError("files must map file names to entries")
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))  # read-only: names stay validated
        for name, entry in self.files.items():
            safe_name(name)
            if not isinstance(entry, FileEntry):
                raise IntegrityError(f"files[{name!r}] must be a FileEntry")
        _names(self.speakers, "speakers")
        _names(self.featured, "featured")
        if not set(self.featured) <= set(self.speakers):
            raise IntegrityError("featured must list only speakers of this voice")

    @classmethod
    def from_dict(cls, data: Any) -> VoiceManifest:
        d = _object(data, MANIFEST, _KEYS)
        files = d["files"]
        if not isinstance(files, dict):
            raise IntegrityError("files must be a JSON object")
        return cls(
            id=d["id"],
            version=d["version"],
            language=d["language"],
            engine=d["engine"],
            sample_rate=d["sample_rate"],
            licence=d["licence"],
            data_licence=d["data_licence"],
            attribution=d["attribution"],
            origin=cast(Origin, d["origin"]),
            files={safe_name(k): FileEntry.from_dict(v, f"files[{k!r}]") for k, v in files.items()},
            speakers=d["speakers"],
            featured=d["featured"],
        )

    @classmethod
    def from_json(cls, path: Path) -> VoiceManifest:
        return cls.from_dict(read_json(path, Path(path).name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "language": self.language,
            "engine": self.engine,
            "sample_rate": self.sample_rate,
            "licence": self.licence,
            "data_licence": self.data_licence,
            "attribution": self.attribution,
            "origin": self.origin,
            "files": {name: entry.to_dict() for name, entry in self.files.items()},
            "speakers": list(self.speakers),
            "featured": list(self.featured),
        }

    def manifest_digest(self) -> str:
        """sha256 of the canonical JSON: sorted keys, no whitespace, ASCII."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


# --- packages -------------------------------------------------------------------------------------------------


def _verify_file(path: Path, name: str, entry: FileEntry) -> None:
    if path.is_symlink():
        raise IntegrityError(f"{name} is a symbolic link, which is refused")
    try:
        st = path.stat()
    except OSError:
        raise IntegrityError(f"{name} is missing") from None
    if not stat.S_ISREG(st.st_mode):
        raise IntegrityError(f"{name} is not a regular file")
    if st.st_size != entry.size:
        raise IntegrityError(f"{name}: size is {st.st_size} bytes, the manifest pins {entry.size}")
    if sha256_file(path) != entry.sha256:
        raise IntegrityError(f"{name}: sha256 does not match the manifest")


def _require_consent(directory: Path, manifest: VoiceManifest) -> None:
    if CARD not in manifest.files:  # pinned, or anyone could swap the statement
        raise ConsentMissing(f"a fine-tuned voice must list {CARD}, with its consent statement, in {MANIFEST}")
    try:
        card = read_json(directory / CARD, CARD)
    except IntegrityError as e:
        raise ConsentMissing(f"{CARD} cannot be read: {e}") from None
    consent = card.get("consent") if isinstance(card, dict) else None
    statement = consent.get("statement") if isinstance(consent, dict) else None
    if not isinstance(statement, str) or not statement.strip():
        raise ConsentMissing(f"{CARD} has no consent statement (consent.statement)")


def verify_package(directory: Path) -> VoiceManifest:
    """The package's manifest, once every listed file matches it (and a fine-tuned voice carries consent)."""
    directory = Path(directory)
    manifest = VoiceManifest.from_json(directory / MANIFEST)
    for name in REQUIRED_FILES:
        if name not in manifest.files:
            raise IntegrityError(f"{MANIFEST} does not list {name}")
    for name, entry in manifest.files.items():
        _verify_file(directory / safe_name(name), name, entry)
    if manifest.origin == "fine-tuned":
        _require_consent(directory, manifest)
    return manifest
