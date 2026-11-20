# SPDX-License-Identifier: MPL-2.0
"""Speech jobs (spec §6.4): one at a time, Stop between sentences, warnings, and the Repair hand-off."""

import io
import logging
import threading
import wave
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from gui_helpers import wait_until
from helpers import TEST_THREADS, TINY

from bytestts import Audio, BytesTTS
from bytestts.errors import IntegrityError, VoiceNotFound
from bytestts.gui.jobs import JobStatus, SpeakRequest, SpeechJobs
from bytestts.gui.session import Busy, CpuSlot

RATE = 22050


def request(text: str = "One. Two. Three.", **changes: Any) -> SpeakRequest:
    base: dict[str, Any] = {
        "text": text, "voice": "tiny", "speaker": None, "speed": 1.0, "device": "cpu", "threads": TEST_THREADS,
    }  # fmt: skip
    return SpeakRequest(**{**base, **changes})


def real_tiny(voice: str, device: str, threads: int) -> BytesTTS:
    return BytesTTS.load_path(TINY, device="cpu", threads=TEST_THREADS)


def frames(wav: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


class FakeTTS:
    """Yields `n` 0.1 s chunks of a tone. After the first, waits for `gate` (if given) before the rest."""

    def __init__(self, n: int = 3, gate: threading.Event | None = None) -> None:
        self.n = n
        self.gate = gate

    def stream(self, text: str, *, speaker: str | None = None, speed: float = 1.0) -> Iterator[Audio]:
        for i in range(self.n):
            if i == 1 and self.gate is not None:
                assert self.gate.wait(10)
            yield Audio(np.full(2205, 0.1, np.float32), RATE)


def jobs_with(loader: Any, damaged: list[str] | None = None) -> tuple[SpeechJobs, CpuSlot]:
    slot = CpuSlot()
    on_damaged = (damaged if damaged is not None else []).append
    return SpeechJobs(slot, on_damaged=on_damaged, loader=loader), slot


def finish(jobs: SpeechJobs, job_id: int) -> JobStatus:
    wait_until(lambda: jobs.status(job_id).state != "running", timeout=60)
    return jobs.status(job_id)


def test_full_job_matches_synthesize_in_structure() -> None:
    jobs, slot = jobs_with(real_tiny)
    job = jobs.start(request())
    status = finish(jobs, job)
    assert (status.state, status.sentences_done, status.error) == ("done", 3, None)
    got = frames(jobs.audio(job))
    reference = BytesTTS.load_path(TINY, device="cpu", threads=TEST_THREADS).synthesize("One. Two. Three.")
    assert len(got) == len(reference.samples)
    lengths = [len(a.samples) for a in real_tiny("tiny", "cpu", 1).stream("One. Two. Three.")]
    pause = round(BytesTTS.SENTENCE_PAUSE_S * RATE)
    ref = frames(reference.to_wav_bytes())
    start = 0
    for n in lengths[:-1]:
        start += n
        assert not got[start : start + pause].any() and not ref[start : start + pause].any()
        start += pause
    assert not slot.busy


def test_stop_after_first_sentence() -> None:
    gate = threading.Event()
    jobs, _ = jobs_with(lambda v, d, t: FakeTTS(3, gate))
    job = jobs.start(request())
    wait_until(lambda: jobs.status(job).sentences_done == 1)
    jobs.stop(job)
    gate.set()
    status = finish(jobs, job)
    assert (status.state, status.sentences_done) == ("stopped", 2)  # the sentence in progress still finishes
    full, _ = jobs_with(lambda v, d, t: FakeTTS(3))
    whole = full.start(request())
    finish(full, whole)
    assert len(frames(jobs.audio(job))) < len(frames(full.audio(whole)))


def test_second_job_while_running_is_busy() -> None:
    gate = threading.Event()
    jobs, slot = jobs_with(lambda v, d, t: FakeTTS(3, gate))
    job = jobs.start(request())
    with pytest.raises(Busy, match="Speak is running"):
        jobs.start(request())
    gate.set()
    assert finish(jobs, job).state == "done"
    assert not slot.busy


def test_errors_are_one_line() -> None:
    jobs, slot = jobs_with(real_tiny)
    status = finish(jobs, jobs.start(request("123")))  # the tiny voice speaks no digits
    assert status.state == "failed" and status.error and "\n" not in status.error
    with pytest.raises(KeyError):
        jobs.audio(status.id)
    assert not slot.busy


@pytest.mark.parametrize(
    "error", [VoiceNotFound("tiny is not cached and offline mode is on"), IntegrityError("voice.json mismatch")]
)
def test_damaged_voice_marks_repair(error: Exception) -> None:
    def broken(v: str, d: str, t: int) -> BytesTTS:
        raise error

    damaged: list[str] = []
    jobs, slot = jobs_with(broken, damaged)
    status = finish(jobs, jobs.start(request()))
    assert status.state == "failed" and status.error is not None and "Repair" in status.error
    assert damaged == ["tiny"]
    assert not slot.busy


def test_warnings_reach_the_status() -> None:
    jobs, _ = jobs_with(real_tiny)
    status = finish(jobs, jobs.start(request("hi ½.")))
    assert status.state == "done"
    assert any("U+00BD" in w for w in status.warnings)
    assert not any("hi" in w for w in status.warnings)  # the text itself is never logged


class WarningTTS(FakeTTS):
    """FakeTTS that logs a library warning from the job's own thread first."""

    def stream(self, text: str, *, speaker: str | None = None, speed: float = 1.0) -> Iterator[Audio]:
        logging.getLogger("bytestts.text").warning("own warning")
        yield from super().stream(text, speaker=speaker, speed=speed)


def test_warnings_from_other_threads_stay_out() -> None:
    """A download or a second panel thread logging meanwhile must not reach this Speak's status line."""
    gate = threading.Event()
    jobs, _ = jobs_with(lambda v, d, t: WarningTTS(3, gate))
    job = jobs.start(request())
    wait_until(lambda: jobs.status(job).sentences_done >= 1)
    other = threading.Thread(target=lambda: logging.getLogger("bytestts.x").warning("elsewhere"))
    other.start()
    other.join(10)
    gate.set()
    status = finish(jobs, job)
    assert status.state == "done"
    assert "own warning" in status.warnings
    assert not any("elsewhere" in w for w in status.warnings)


def test_loaded_voice_is_reused_until_settings_change() -> None:
    loads: list[tuple[str, str, int]] = []

    def counting(v: str, d: str, t: int) -> FakeTTS:
        loads.append((v, d, t))
        return FakeTTS(1)

    jobs, _ = jobs_with(counting)
    for changes in ({}, {}, {"threads": 2}, {"threads": 2}, {"voice": "other"}):
        finish(jobs, jobs.start(request(**changes)))
    assert loads == [("tiny", "cpu", 1), ("tiny", "cpu", 2), ("other", "cpu", 1)]


def test_only_the_last_job_is_kept() -> None:
    jobs, _ = jobs_with(lambda v, d, t: FakeTTS(1))
    first = jobs.start(request())
    finish(jobs, first)
    second = jobs.start(request())
    finish(jobs, second)
    assert second != first
    with pytest.raises(KeyError):
        jobs.status(first)
    with pytest.raises(KeyError):
        jobs.stop(first)
    with pytest.raises(KeyError):
        jobs.audio(first)
