# SPDX-License-Identifier: MPL-2.0
"""The panel server, part 2 (spec §6.3, §6.10): Speak, Stop, audio, WAV export and Quit."""

import asyncio
import io
import tempfile
import threading
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pytest
from gui_helpers import CODE, PORT, SECRET, api_client, install, tiny_manifest, use_catalogue, wait_until
from helpers import TEST_THREADS, TINY
from starlette.testclient import TestClient

from bytestts import Audio, BytesTTS
from bytestts.errors import VoiceNotFound
from bytestts.gui import saving
from bytestts.gui.protection import LaunchCode
from bytestts.gui.saving import SaveTickets
from bytestts.gui.server import create_app
from bytestts.gui.session import Session

RATE = 22050


def real_tiny(voice: str, device: str, threads: int) -> BytesTTS:
    return BytesTTS.load_path(TINY, device="cpu", threads=TEST_THREADS)


class FakeTTS:
    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate

    def stream(self, text: str, *, speaker: str | None = None, speed: float = 1.0) -> Iterator[Audio]:
        if self.gate is not None:
            assert self.gate.wait(10)
        yield Audio(np.full(2205, 0.1, np.float32), RATE)


def panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    loader: Any = real_tiny,
    voices: dict[str, Any] | None = None,
) -> tuple[TestClient, Session]:
    catalogue = voices if voices is not None else {"tiny": tiny_manifest()}
    use_catalogue(monkeypatch, tmp_path, catalogue)
    for m in catalogue.values():
        install(m)
    session = Session(SaveTickets())
    app = create_app(
        port=PORT, secret=SECRET, launch_code=LaunchCode(CODE), max_threads=2, session=session, loader=loader
    )
    return api_client(app), session


def body(**changes: Any) -> dict[str, Any]:
    base = {"text": "One. Two.", "voice": "tiny", "speaker": None, "speed": 1.0, "device": "cpu", "threads": 1}
    return {**base, **changes}


def finish(client: TestClient, job: int) -> dict[str, Any]:
    wait_until(lambda: client.get(f"/api/jobs/{job}").json()["state"] != "running", timeout=60)
    result: dict[str, Any] = client.get(f"/api/jobs/{job}").json()
    return result


def wav_bytes(frames: int = 2000, channels: int = 1, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.arange(frames * channels, dtype="<i2").tobytes())
    return buf.getvalue()


def frames_of(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data)) as w:
        return w.readframes(w.getnframes())


def test_speak_and_fetch_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _ = panel(monkeypatch, tmp_path)
    started = client.post("/api/jobs", json=body(speed=1))  # JS sends 1.0 as 1: an int must pass for speed
    assert started.status_code == 201, started.json()
    job = started.json()["id"]
    status = finish(client, job)
    assert (status["id"], status["state"], status["sentences_done"], status["error"]) == (job, "done", 2, None)
    assert status["warnings"] == []
    audio = client.get(f"/api/jobs/{job}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(audio.content)) as w:
        assert (w.getnchannels(), w.getframerate()) == (1, RATE) and w.getnframes() > 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"extra": 1}, "extra"),
        ({"text": ""}, "text must be 1 to 5000 characters"),
        ({"text": "a" * 5001}, "text must be 1 to 5000 characters"),
        ({"speed": 0.49}, "speed must be between 0.5 and 2.0"),
        ({"speed": 2.01}, "speed must be between 0.5 and 2.0"),
        ({"threads": 0}, "threads must be 1 to 2"),
        ({"threads": 3}, "threads must be 1 to 2"),
        ({"threads": "1"}, "threads"),
        ({"device": "auto"}, "device"),
        ({"voice": "nope"}, "unknown voice nope"),
    ],
)
def test_job_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changes: dict[str, Any], message: str) -> None:
    client, _ = panel(monkeypatch, tmp_path)
    refused = client.post("/api/jobs", json=body(**changes))
    assert refused.status_code == 400
    error = refused.json()["error"]
    assert message in error and "\n" not in error


def test_voice_must_be_downloaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_catalogue(monkeypatch, tmp_path, {"tiny": tiny_manifest()})  # nothing installed
    session = Session(SaveTickets())
    client = api_client(
        create_app(port=PORT, secret=SECRET, launch_code=LaunchCode(CODE), max_threads=2, session=session)
    )
    refused = client.post("/api/jobs", json=body())
    assert refused.status_code == 400 and refused.json()["error"] == "voice tiny is not downloaded"


def test_text_limit_counts_code_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _ = panel(monkeypatch, tmp_path, loader=lambda v, d, t: FakeTTS())
    assert client.post("/api/jobs", json=body(text="\U0001f600" * 5000)).status_code == 201
    wait_until(lambda: not client.get("/api/status").json()["busy"])
    refused = client.post("/api/jobs", json=body(text="\U0001f600" * 5001))
    assert refused.status_code == 400 and "5000" in refused.json()["error"]


def test_speaker_rules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    voices = {"tiny": tiny_manifest(), "multi": tiny_manifest(id="multi", speakers=["p1"])}
    client, _ = panel(monkeypatch, tmp_path, loader=lambda v, d, t: FakeTTS(), voices=voices)
    refused = client.post("/api/jobs", json=body(speaker="x"))
    assert refused.status_code == 400 and refused.json()["error"] == "voice tiny has no speakers"
    for speaker in (None, "p1"):
        started = client.post("/api/jobs", json=body(voice="multi", speaker=speaker))
        assert started.status_code == 201, started.json()
        finish(client, started.json()["id"])
    refused = client.post("/api/jobs", json=body(voice="multi", speaker="p2"))
    assert refused.status_code == 400 and refused.json()["error"] == "speaker p2 is not in voice multi"


def test_stop_after_done_is_harmless(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _ = panel(monkeypatch, tmp_path, loader=lambda v, d, t: FakeTTS())
    job = client.post("/api/jobs", json=body()).json()["id"]
    assert finish(client, job)["state"] == "done"
    stopped = client.post(f"/api/jobs/{job}/stop")
    assert stopped.status_code == 200 and stopped.json() == {"state": "done"}
    assert client.get(f"/api/jobs/{job}").json()["state"] == "done"
    for response in (client.post("/api/jobs/999/stop"), client.get("/api/jobs/999"), client.get("/api/jobs/999/audio")):
        assert response.status_code == 404 and response.json()["error"]


def test_second_cpu_task_is_409(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gate = threading.Event()
    client, _ = panel(monkeypatch, tmp_path, loader=lambda v, d, t: FakeTTS(gate))
    job = client.post("/api/jobs", json=body()).json()["id"]
    assert client.get("/api/status").json()["busy"] is True
    assert client.post("/api/jobs", json=body()).status_code == 409
    export = client.post("/api/export?format=wav&quality=standard", content=wav_bytes())
    assert export.status_code == 409 and "Speak is running" in export.json()["error"]
    gate.set()
    assert finish(client, job)["state"] == "done"
    assert client.post("/api/export?format=wav&quality=standard", content=wav_bytes()).status_code == 200


def test_damaged_voice_shows_needs_repair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken(v: str, d: str, t: int) -> BytesTTS:
        raise VoiceNotFound("tiny is not cached and offline mode is on")

    client, _ = panel(monkeypatch, tmp_path, loader=broken)
    status = finish(client, client.post("/api/jobs", json=body()).json()["id"])
    assert status["state"] == "failed" and "Repair" in status["error"]
    assert {v["id"]: v["state"] for v in client.get("/api/voices").json()}["tiny"] == "needs-repair"
    refused = client.post("/api/jobs", json=body())
    assert refused.status_code == 400 and refused.json()["error"] == "voice tiny needs repair"


def test_cuda_without_provider_fails_one_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    client, _ = panel(monkeypatch, tmp_path, loader=None)  # the real load_voice
    status = finish(client, client.post("/api/jobs", json=body(device="cuda")).json()["id"])
    assert status["state"] == "failed" and "CUDA" in status["error"] and "\n" not in status["error"]


def test_export_wav_without_ticket_returns_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _ = panel(monkeypatch, tmp_path)
    data = wav_bytes(frames=5000, channels=2, rate=48000)
    exported = client.post("/api/export?format=wav&quality=standard", content=data)
    assert exported.status_code == 200 and exported.headers["content-type"] == "audio/wav"
    assert frames_of(exported.content) == frames_of(data)


def test_export_with_ticket_writes_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, session = panel(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "out.wav"
    ticket = session.tickets.issue(target)
    data = wav_bytes()
    saved = client.post(f"/api/export?format=wav&quality=standard&ticket={ticket}", content=data)
    assert saved.status_code == 200 and saved.json() == {"saved": "out.wav"}
    assert frames_of(target.read_bytes()) == frames_of(data)
    assert sorted(p.name for p in out_dir.iterdir()) == ["out.wav"]
    again = client.post(f"/api/export?format=wav&quality=standard&ticket={ticket}", content=data)
    assert again.status_code == 400 and "ticket" in again.json()["error"]


async def _raw_post(app: Any, path: str, query: str, headers: list[tuple[bytes, bytes]], content: bytes) -> int:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(b"host", f"127.0.0.1:{PORT}".encode()), (b"authorization", f"Bearer {SECRET}".encode()), *headers],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", PORT),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": content, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return int(next(m["status"] for m in sent if m["type"] == "http.response.start"))


def test_export_refusals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _ = panel(monkeypatch, tmp_path)
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    for query in ("format=mp3&quality=standard", "format=wav&quality=best", "quality=standard"):
        refused = client.post(f"/api/export?{query}", content=wav_bytes())
        assert refused.status_code == 400 and "\n" not in refused.json()["error"], query
    not_wav = client.post("/api/export?format=wav&quality=standard", content=b"not a wav file at all")
    assert not_wav.status_code == 400 and "WAV" in not_wav.json()["error"]
    forged = [(b"content-length", str(saving.BYTE_CAP + 1).encode())]
    assert asyncio.run(_raw_post(client.app, "/api/export", "format=wav&quality=standard", forged, b"x")) == 413
    monkeypatch.setattr(saving, "BYTE_CAP", 10_000)

    def stream() -> Iterator[bytes]:
        for _ in range(20):
            yield b"x" * 1000

    too_big = client.post("/api/export?format=wav&quality=standard", content=stream())
    assert too_big.status_code == 413
    assert list(temp.iterdir()) == []  # every temporary file is gone


def test_quit_requests_quit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, session = panel(monkeypatch, tmp_path)
    assert client.post("/api/quit").json() == {"ok": True}
    assert session.quit.is_set()
