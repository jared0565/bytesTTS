# SPDX-License-Identifier: MPL-2.0
"""Verified, atomic downloads into the voice cache.

Every file is pinned by size and sha256 in the catalogue. A download goes to a unique `.part` file beside its
destination, is verified, then renamed into place; on any failure the `.part` file is deleted. Redirects are
followed by hand so that every hop is checked before it is requested: https only, with plain http allowed only to
127.0.0.1 or localhost (local test servers and mirrors). A hop that came from https may never go back to http, nor
reach localhost or a non-global IP address, so a hostile redirect cannot send a request to a local or private
service. A redirect to an https *hostname* that resolves to an internal address is not blocked (that would need
DNS pinning); TLS then requires that host to present a valid certificate for the name. Archives are never extracted.
"""

from __future__ import annotations

import ipaddress
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urljoin, urlsplit

import requests

from bytestts.catalog import catalog
from bytestts.catalog.cache import voice_dir
from bytestts.catalog.manifest import MANIFEST, VoiceManifest, safe_name, sha256_file, verify_package
from bytestts.errors import IntegrityError, VoiceNotFound

MAX_REDIRECTS = 5
_LOOPBACK = ("127.0.0.1", "localhost")
_CHUNK = 1 << 16
_REPLACE_ATTEMPTS = 20  # about 2 s of retries in all
# identity: the size cap then counts the bytes on the wire, and no compressed body can inflate behind it
_HEADERS = {"Accept-Encoding": "identity"}

Progress = Callable[[int, int], None]


def _is_internal(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return not ipaddress.ip_address(host).is_global  # loopback, private, link-local, unspecified, mapped
    except ValueError:
        return False  # a name: not resolved here


def _check_url(url: str, *, after: str | None = None) -> None:
    """Refuse `url` unless it may be requested; `after` is the URL that redirected to it, if any."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    from_https = after is not None and urlsplit(after).scheme == "https"
    if parts.scheme == "https" and host and not (from_https and _is_internal(host)):
        return
    if parts.scheme == "http" and host in _LOOPBACK and not from_https:
        return
    raise ValueError(
        f"refusing to download {url!r}: https only (plain http only to 127.0.0.1 or localhost)"
        + (f", and a redirect from https may not go to http or inward (from {after!r})" if from_https else "")
    )


def _verified(path: Path, sha256: str, size: int) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size == size and sha256_file(path) == sha256
    except OSError:
        return False


def _replace_into(part: Path, dest: Path, already_right: Callable[[], bool]) -> None:
    """Rename `part` over `dest`. Windows refuses while another thread or process has `dest` open or is replacing
    it (measured: 24 of 240 concurrent downloads failed without the retry). Then `dest` is fine if it is already
    right; otherwise the state is transient, so retry briefly."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(part, dest)
            return
        except PermissionError:
            if already_right():
                return
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.01 * (attempt + 1))


def _fetch(url: str, out: BinaryIO, *, size: int, timeout: tuple[float, float], progress: Progress | None) -> None:
    previous: str | None = None
    with requests.Session() as session:
        for _ in range(MAX_REDIRECTS + 1):
            _check_url(url, after=previous)  # every hop, before it is requested
            with session.get(url, stream=True, allow_redirects=False, timeout=timeout, headers=_HEADERS) as r:
                if r.is_redirect:
                    previous, url = url, urljoin(url, r.headers["Location"])
                    continue
                r.raise_for_status()
                announced = r.headers.get("Content-Length", "")
                if announced.isdigit() and int(announced) > size:
                    raise IntegrityError(f"{url} announces {announced} bytes, more than the pinned {size}")
                done = 0
                for chunk in r.iter_content(_CHUNK):
                    done += len(chunk)
                    if done > size:
                        raise IntegrityError(f"{url} sent more than the pinned {size} bytes")
                    out.write(chunk)
                    if progress is not None:
                        progress(done, size)
                return
    raise requests.TooManyRedirects(f"more than {MAX_REDIRECTS} redirects, last to {url}")


def download_file(
    url: str,
    dest: Path,
    *,
    sha256: str,
    size: int,
    timeout: tuple[float, float] = (10.0, 60.0),
    progress: Progress | None = None,
) -> Path:
    """`dest`, holding exactly the pinned bytes: reused if it already does, else downloaded and verified."""
    dest = Path(dest)
    _check_url(url)
    if _verified(dest, sha256, size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")  # unique per thread
    part = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            _fetch(url, out, size=size, timeout=timeout, progress=progress)
        if not _verified(part, sha256, size):
            raise IntegrityError(f"{dest.name}: the download does not match its pinned size and sha256")
        _replace_into(part, dest, lambda: _verified(dest, sha256, size))
        return dest
    finally:
        part.unlink(missing_ok=True)  # gone already after a successful replace


def _manifest_is(path: Path, m: VoiceManifest) -> bool:
    try:
        return VoiceManifest.from_json(path) == m
    except IntegrityError:
        return False


def _write_manifest(directory: Path, m: VoiceManifest) -> None:
    """voice.json from the catalogue entry, so the cached voice is a package verify_package accepts."""
    path = directory / MANIFEST
    if _manifest_is(path, m):
        return
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=directory, prefix=MANIFEST + ".", suffix=".part")
    part = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write((json.dumps(m.to_dict(), indent=2, sort_keys=True) + "\n").encode("ascii"))
        _replace_into(part, path, lambda: _manifest_is(path, m))
    finally:
        part.unlink(missing_ok=True)


def ensure_voice(voice_id: str, *, offline: bool = False, progress: Progress | None = None) -> Path:
    """A verified package directory for a catalogue voice, downloading what is missing or wrong."""
    manifest = catalog.load_catalog().get(voice_id)
    if manifest is None:
        raise VoiceNotFound(f"unknown voice {voice_id}")
    directory = voice_dir(manifest)
    missing = {
        name: entry
        for name, entry in manifest.files.items()
        if not _verified(directory / safe_name(name), entry.sha256, entry.size)
    }
    if missing and offline:
        raise VoiceNotFound(f"{voice_id} is not cached and offline mode is on")
    for name, entry in missing.items():
        if entry.url is None:
            raise IntegrityError(f"voice {voice_id}: {name} has no download URL")
        download_file(entry.url, directory / safe_name(name), sha256=entry.sha256, size=entry.size, progress=progress)
    _write_manifest(directory, manifest)  # the catalogue is the authority: a differing cached copy is replaced
    if verify_package(directory) != manifest:
        raise IntegrityError(f"{directory / MANIFEST} does not match the catalogue")
    return directory
