# SPDX-License-Identifier: MPL-2.0
"""Saving: one-use tickets for paths the user chose in the Save dialog, file names, and the WAV checks.

The page never sends a path. In the app window it asks the bridge for a ticket; the path the user picked stays
here, under the ticket, and the export call names only the ticket (spec §6.7, §6.10). Audio that reaches the
server is read with the standard library's `wave` only, never with a decoder library.
"""

from __future__ import annotations

import os
import re
import secrets
import struct
import threading
import time
import wave
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

SAMPLE_LIMIT = 172_800_000  # every channel counted: 60 min of mono or 30 min of stereo at 48 kHz (spec §6.9)
HEADER_ALLOWANCE = 4096
BYTE_CAP = SAMPLE_LIMIT * 2 + HEADER_ALLOWANCE
TICKET_TTL_S = 600.0
MIN_RATE, MAX_RATE = 8000, 48000
_BLOCK_FRAMES = 65_536
_REPLACE_ATTEMPTS = 20  # about 2 s of retries, as catalog/download.py does for Windows sharing errors

_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = re.compile(r"(con|prn|aux|nul|com[0-9]|lpt[0-9])", re.IGNORECASE)  # Windows device names
_MAX_NAME = 200


class TicketError(ValueError):
    """A save ticket that is unknown, already used, or expired."""


class SaveTickets:
    """Paths the user chose, each behind a random one-use ticket that expires after TICKET_TTL_S."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._paths: dict[str, tuple[Path, float]] = {}

    def issue(self, path: Path) -> str:
        ticket = secrets.token_urlsafe(16)
        with self._lock:
            self._paths[ticket] = (Path(path), self._clock() + TICKET_TTL_S)
        return ticket

    def redeem(self, ticket: str) -> Path:
        with self._lock:
            entry = self._paths.pop(ticket, None)
        if entry is None:
            raise TicketError("unknown or already used save ticket: choose where to save again")
        path, expires = entry
        if self._clock() > expires:
            raise TicketError("the save ticket expired: choose where to save again")
        return path


def safe_file_name(name: str) -> str:
    """`name` as a Windows-safe file name without extension: forbidden characters become "-", trailing dots and
    spaces go, a device name gets "-recording", and an empty result is "recording"."""
    cleaned = _FORBIDDEN.sub("-", name).strip(" ").rstrip(". ")[:_MAX_NAME].rstrip(". ")
    if not cleaned.strip("."):
        return "recording"
    stem = cleaned.split(".")[0]
    if _RESERVED.fullmatch(stem):  # Windows reserves the part before the first dot: "con.txt" opens the console
        return f"{stem}-recording{cleaned[len(stem) :]}"
    return cleaned


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_rate: int
    frames: int


class BadAudio(ValueError):
    """Audio the panel will not read: not a PCM 16-bit WAV within the limits."""


def check_wav(path: Path) -> WavInfo:
    """The WAV's shape, if it is PCM 16-bit, 1 or 2 channels, 8 to 48 kHz and within SAMPLE_LIMIT."""
    try:
        with wave.open(str(path), "rb") as w:
            info = WavInfo(w.getnchannels(), w.getframerate(), w.getnframes())
            width = w.getsampwidth()
    except (wave.Error, EOFError, struct.error) as e:
        raise BadAudio(f"not a PCM WAV file ({e})") from None
    if width != 2:
        raise BadAudio(f"only 16-bit WAV is accepted, not {8 * width}-bit")
    if info.channels not in (1, 2):
        raise BadAudio(f"only mono or stereo WAV is accepted, not {info.channels} channels")
    if not MIN_RATE <= info.sample_rate <= MAX_RATE:
        raise BadAudio(f"the sample rate must be {MIN_RATE} to {MAX_RATE} Hz, not {info.sample_rate}")
    if info.frames * info.channels > SAMPLE_LIMIT:
        raise BadAudio("the audio is too long: the limit is 60 minutes of mono or 30 of stereo at 48 kHz")
    return info


def copy_wav(src: Path, dest: Path) -> None:
    """`src` (checked by check_wav) rewritten as a fresh PCM 16-bit WAV at `dest`, in blocks."""
    with wave.open(str(src), "rb") as r, wave.open(str(dest), "wb") as w:
        w.setnchannels(r.getnchannels())
        w.setsampwidth(2)
        w.setframerate(r.getframerate())
        while block := r.readframes(_BLOCK_FRAMES):
            w.writeframes(block)


def replace_into_place(tmp: Path, target: Path) -> None:
    """Move `tmp` over `target`, retrying briefly while Windows reports a sharing violation."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.01 * (attempt + 1))


class BodyTooLarge(ValueError):
    """A request body larger than its cap."""


async def receive_body(chunks: AsyncIterator[bytes], out: BinaryIO, *, cap: int) -> int:
    """Write the body to `out`, refusing (before writing) any chunk that would take it past `cap` bytes."""
    total = 0
    async for chunk in chunks:
        if total + len(chunk) > cap:
            raise BodyTooLarge(f"the audio is larger than {cap} bytes")
        out.write(chunk)
        total += len(chunk)
    return total
