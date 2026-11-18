# SPDX-License-Identifier: MPL-2.0
"""Voice downloads for the panel (spec §6.5): one at a time, with progress across files, Cancel and Repair.

"Downloaded" is a cheap check: every file in the cache at its manifest size. The full hash check still happens
whenever a voice is loaded; when that load fails, the job marks the voice damaged, and Repair is the ordinary
download, which keeps the files whose hash matches and fetches the rest.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from bytestts.catalog import catalog, download
from bytestts.catalog.cache import voice_dir
from bytestts.catalog.manifest import VoiceManifest, safe_name
from bytestts.errors import VoiceNotFound
from bytestts.gui.session import Busy, one_line

VoiceState = Literal["missing", "downloaded", "needs-repair"]
DownloadState = Literal["idle", "running", "done", "cancelled", "failed"]


class DownloadCancelled(Exception):
    """Raised from the progress callback when the user presses Cancel."""


@dataclass(frozen=True)
class DownloadStatus:
    voice: str
    state: DownloadState
    done: int
    total: int
    error: str | None


def _manifest(voice_id: str) -> VoiceManifest:
    m = catalog.load_catalog().get(voice_id)
    if m is None:
        raise VoiceNotFound(f"unknown voice {voice_id}")
    return m


def files_present(m: VoiceManifest) -> bool:
    """Every file of the voice is in the cache at its manifest size (not hashed: see the module docstring)."""
    d = voice_dir(m)
    for name, entry in m.files.items():
        try:
            if (d / safe_name(name)).stat().st_size != entry.size:
                return False
        except OSError:
            return False
    return True


class _Progress:
    """ensure_voice reports (bytes done, size) per file. A file is finished when its last report had done == size,
    so the next report starts a new file; the sum runs across files, capped at the voice's total."""

    def __init__(self, total: int, cancel: threading.Event) -> None:
        self.total = total
        self._cancel = cancel
        self._finished = 0
        self._done = 0
        self._size: int | None = None

    def __call__(self, done: int, size: int) -> None:
        if self._cancel.is_set():
            raise DownloadCancelled("download cancelled")
        if self._size is not None and self._done == self._size:
            self._finished += self._size
        self._done, self._size = done, size

    @property
    def value(self) -> int:
        return min(self._finished + self._done, self.total)


@dataclass
class _Job:
    progress: _Progress
    cancel: threading.Event
    state: DownloadState = "running"
    error: str | None = None


class Downloads:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._damaged: set[str] = set()
        self._jobs: dict[str, _Job] = {}
        self._running: str | None = None

    def state(self, voice_id: str) -> VoiceState:
        m = _manifest(voice_id)
        with self._lock:
            if voice_id in self._damaged:
                return "needs-repair"
        return "downloaded" if files_present(m) else "missing"

    def mark_damaged(self, voice_id: str) -> None:
        with self._lock:
            self._damaged.add(voice_id)

    def start(self, voice_id: str) -> None:
        m = _manifest(voice_id)
        total = sum(entry.size for entry in m.files.values())
        cancel = threading.Event()
        job = _Job(_Progress(total, cancel), cancel)
        with self._lock:
            if self._running is not None:
                raise Busy(f"the download of {self._running} is running")
            self._running = voice_id
            self._jobs[voice_id] = job
        threading.Thread(target=self._run, args=(voice_id, job), daemon=True, name=f"download {voice_id}").start()

    def _run(self, voice_id: str, job: _Job) -> None:
        state: DownloadState
        error: str | None = None
        try:
            download.ensure_voice(voice_id, progress=job.progress)
            state = "done"
        except DownloadCancelled:
            state = "cancelled"
        except Exception as e:  # any failure is reported as one line on the page
            state, error = "failed", one_line(e)
        with self._lock:
            if state == "done":
                self._damaged.discard(voice_id)
            job.state, job.error = state, error
            self._running = None

    def status(self, voice_id: str) -> DownloadStatus:
        m = _manifest(voice_id)
        with self._lock:
            job = self._jobs.get(voice_id)
            if job is None:
                total = sum(entry.size for entry in m.files.values())
                return DownloadStatus(voice_id, "idle", 0, total, None)
            return DownloadStatus(voice_id, job.state, job.progress.value, job.progress.total, job.error)

    def cancel(self, voice_id: str) -> None:
        with self._lock:
            job = self._jobs.get(voice_id)
            if job is not None and job.state == "running":
                job.cancel.set()
