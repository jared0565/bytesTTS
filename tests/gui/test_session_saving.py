# SPDX-License-Identifier: MPL-2.0
"""The CPU slot, the session, save tickets and the WAV checks (spec §6.2 rule 9, §6.7, §6.10)."""

import asyncio
import string
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from bytestts.gui import saving
from bytestts.gui.saving import (
    BadAudio,
    BodyTooLarge,
    SaveTickets,
    TicketError,
    WavInfo,
    check_wav,
    copy_wav,
    receive_body,
    replace_into_place,
    safe_file_name,
)
from bytestts.gui.session import Busy, CpuSlot, Session, one_line


def write_wav(path: Path, *, rate: int = 22050, channels: int = 1, width: int = 2, frames: int = 100) -> Path:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=frames * channels * width, dtype=np.uint8).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(data)
    return path


def test_slot_allows_one_task() -> None:
    slot = CpuSlot()
    assert not slot.busy
    slot.acquire("Speak")
    assert slot.busy
    with pytest.raises(Busy, match="Speak is running"):
        slot.acquire("export")
    slot.release()
    assert not slot.busy
    slot.acquire("export")
    assert slot.busy


def test_ticket_works_once_and_expires(tmp_path: Path) -> None:
    now = [100.0]
    tickets = SaveTickets(clock=lambda: now[0])
    p = tmp_path / "a.wav"
    t = tickets.issue(p)
    assert len(t) >= 22
    assert tickets.redeem(t) == p
    with pytest.raises(TicketError):
        tickets.redeem(t)  # used
    late = tickets.issue(p)
    now[0] += 600.1
    with pytest.raises(TicketError):
        tickets.redeem(late)  # expired
    with pytest.raises(TicketError):
        tickets.redeem("nope")
    fresh = tickets.issue(p)
    now[0] += 599.9
    assert tickets.redeem(fresh) == p


def test_tickets_are_distinct(tmp_path: Path) -> None:
    tickets = SaveTickets()
    assert len({tickets.issue(tmp_path / "x.wav") for _ in range(50)}) == 50


def test_safe_file_name() -> None:
    assert safe_file_name("es-css10 15:02:07") == "es-css10 15-02-07"
    cleaned = safe_file_name('a<b>"c|d?e*f/g\\h')
    assert not set('<>:"/\\|?*') & set(cleaned) and cleaned.startswith("a")
    assert safe_file_name("tab\there\x00x") == "tab-here-x"
    assert safe_file_name("café ") == "café"
    assert safe_file_name("café.") == "café"
    assert safe_file_name("") == "recording"
    assert safe_file_name("...") == "recording"
    assert safe_file_name("CON") == "CON-recording"
    assert safe_file_name("lpt1") == "lpt1-recording"
    assert safe_file_name("con.txt") == "con-recording.txt"  # Windows reserves the part before the first dot
    assert safe_file_name("console") == "console"
    assert len(safe_file_name("x" * 500)) <= 200


@pytest.mark.parametrize(("rate", "channels"), [(8000, 1), (22050, 1), (48000, 2), (22050, 2)])
def test_check_wav_accepts_pcm16(tmp_path: Path, rate: int, channels: int) -> None:
    path = write_wav(tmp_path / "ok.wav", rate=rate, channels=channels, frames=321)
    assert check_wav(path) == WavInfo(channels=channels, sample_rate=rate, frames=321)


def test_check_wav_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    junk = tmp_path / "junk.wav"
    junk.write_bytes(np.random.default_rng(1).integers(0, 256, 300, dtype=np.uint8).tobytes())
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    bad = [
        junk,
        empty,
        write_wav(tmp_path / "8bit.wav", width=1),
        write_wav(tmp_path / "3ch.wav", channels=3),
        write_wav(tmp_path / "96k.wav", rate=96000),
        write_wav(tmp_path / "low.wav", rate=7999),
    ]
    for path in bad:
        with pytest.raises(BadAudio) as e:
            check_wav(path)
        assert str(e.value) and "\n" not in str(e.value), path.name
    monkeypatch.setattr(saving, "SAMPLE_LIMIT", 1000)
    with pytest.raises(BadAudio, match="too long"):
        check_wav(write_wav(tmp_path / "long.wav", channels=2, frames=501))
    assert check_wav(write_wav(tmp_path / "edge.wav", channels=2, frames=500)).frames == 500


def test_copy_wav_round_trips(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "in.wav", rate=24000, channels=2, frames=70_000)  # more than one block
    dest = tmp_path / "out.wav"
    copy_wav(src, dest)
    with wave.open(str(src)) as a, wave.open(str(dest)) as b:
        assert a.getparams() == b.getparams()
        assert a.readframes(a.getnframes()) == b.readframes(b.getnframes())


def _chunks(n: int, size: int) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        for _ in range(n):
            yield b"x" * size

    return gen()


def test_receive_body_stops_at_the_cap(tmp_path: Path) -> None:
    with (tmp_path / "a").open("wb") as out, pytest.raises(BodyTooLarge):
        asyncio.run(receive_body(_chunks(10, 1000), out, cap=9_999))
    assert (tmp_path / "a").stat().st_size <= 10_000
    with (tmp_path / "b").open("wb") as out:
        assert asyncio.run(receive_body(_chunks(10, 1000), out, cap=10_000)) == 10_000
    assert (tmp_path / "b").stat().st_size == 10_000


def test_replace_into_place_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "out.wav"
    target.write_bytes(b"old")
    tmp = tmp_path / "out.wav.part"
    tmp.write_bytes(b"new")
    replace_into_place(tmp, target)
    assert target.read_bytes() == b"new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.wav"]


def test_request_quit_sets_flags_and_closes_window() -> None:
    closed: list[bool] = []
    session = Session(tickets=SaveTickets())
    session.close_window = lambda: closed.append(True)
    assert not session.quit.is_set() and not session.quitting and not session.unsaved
    session.request_quit()
    assert session.quit.is_set() and session.quitting and closed == [True]


def test_request_quit_without_a_window() -> None:
    session = Session(tickets=SaveTickets())
    session.request_quit()
    assert session.quit.is_set()


def test_one_line() -> None:
    assert one_line(ValueError("a\n  b\n\n c")) == "a b c"
    assert one_line(ValueError("")) == "ValueError"
    assert set(one_line(RuntimeError("x\ty"))) <= set(string.printable)
