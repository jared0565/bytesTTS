# SPDX-License-Identifier: MPL-2.0
"""Speech jobs for the panel (spec §6.4): one at a time, rendered sentence by sentence so Stop can take effect
between sentences. The last loaded voice stays loaded until the voice, device or threads change."""

from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from bytestts.api import BytesTTS
from bytestts.audio import Audio, concat
from bytestts.errors import IntegrityError, VoiceNotFound
from bytestts.gui.session import CpuSlot, one_line

Device = Literal["cpu", "cuda"]
JobState = Literal["running", "done", "stopped", "failed"]
Loader = Callable[[str, Device, int], BytesTTS]


def load_voice(voice_id: str, device: Device, threads: int) -> BytesTTS:
    """A downloaded catalogue voice, verified, never downloaded here (the page offers Speak only once it is)."""
    return BytesTTS.load(voice_id, device=device, threads=threads, offline=True)


@dataclass(frozen=True)
class SpeakRequest:
    text: str
    voice: str
    speaker: str | None
    speed: float
    device: Device
    threads: int


@dataclass(frozen=True)
class JobStatus:
    id: int
    state: JobState
    sentences_done: int
    error: str | None
    warnings: tuple[str, ...]


@dataclass
class _Job:
    id: int
    state: JobState = "running"
    sentences_done: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)
    audio: Audio | None = None


class _Collect(logging.Handler):
    """The library's warnings during a job (such as a character a voice can't say), for the status line. The
    library never logs the text itself. Only the job's own thread counts: the library logger is shared, so a
    download or another request logging meanwhile stays out of this Speak's status."""

    def __init__(self, into: list[str]) -> None:
        super().__init__(logging.WARNING)
        self._into = into
        self._thread = threading.get_ident()  # made on the job's thread, in _run

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self._thread:
            self._into.append(" ".join(record.getMessage().split()))


class SpeechJobs:
    def __init__(self, slot: CpuSlot, *, on_damaged: Callable[[str], None], loader: Loader = load_voice) -> None:
        self._slot = slot
        self._on_damaged = on_damaged
        self._loader = loader
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._job: _Job | None = None
        self._loaded: tuple[tuple[str, Device, int], BytesTTS] | None = None

    def start(self, request: SpeakRequest) -> int:
        """A new job's id. Raises Busy while another CPU task runs. The previous job, and its audio, are gone."""
        self._slot.acquire("Speak")
        try:
            job = _Job(next(self._ids))
            with self._lock:
                self._job = job
            threading.Thread(target=self._run, args=(job, request), daemon=True, name=f"speak {job.id}").start()
        except BaseException:
            self._slot.release()
            raise
        return job.id

    def _voice(self, request: SpeakRequest) -> BytesTTS:
        key = (request.voice, request.device, request.threads)
        if self._loaded is None or self._loaded[0] != key:
            self._loaded = None  # free the old voice before loading the next
            self._loaded = (key, self._loader(*key))
        return self._loaded[1]

    def _run(self, job: _Job, request: SpeakRequest) -> None:
        collect = _Collect(job.warnings)
        library = logging.getLogger("bytestts")
        library.addHandler(collect)
        try:
            try:
                tts = self._voice(request)
            except (VoiceNotFound, IntegrityError) as e:
                self._on_damaged(request.voice)
                raise RuntimeError(f"voice {request.voice} failed its check; press Repair ({one_line(e)})") from e
            chunks: list[Audio] = []
            for chunk in tts.stream(request.text, speaker=request.speaker, speed=request.speed):
                chunks.append(chunk)
                with self._lock:
                    job.sentences_done = len(chunks)
                if job.stop.is_set():
                    break
            audio = concat(chunks, pause_s=BytesTTS.SENTENCE_PAUSE_S)
            with self._lock:
                job.audio = audio
                job.state = "stopped" if job.stop.is_set() else "done"
        except Exception as e:  # any failure reaches the page as one line
            with self._lock:
                job.state, job.error = "failed", one_line(e)
        finally:
            library.removeHandler(collect)
            self._slot.release()

    def _get(self, job_id: int) -> _Job:
        job = self._job
        if job is None or job.id != job_id:
            raise KeyError(job_id)
        return job

    def status(self, job_id: int) -> JobStatus:
        with self._lock:
            job = self._get(job_id)
            return JobStatus(job.id, job.state, job.sentences_done, job.error, tuple(job.warnings))

    def stop(self, job_id: int) -> None:
        with self._lock:
            self._get(job_id).stop.set()

    def audio(self, job_id: int) -> bytes:
        with self._lock:
            audio = self._get(job_id).audio
        if audio is None:
            raise KeyError(job_id)
        return audio.to_wav_bytes()
