# SPDX-License-Identifier: MPL-2.0
"""The window's Save dialog and unsaved flag, as server routes (design B, spec §6.7 as revised 2026-11-18).

The page never gets a path: the route opens the dialog through the session's hook and answers with a one-use
ticket and the chosen file's name.
"""

import threading
from pathlib import Path

from gui_helpers import PORT, SECRET, api_client, wait_until
from starlette.testclient import TestClient

from bytestts.gui.saving import SaveTickets
from bytestts.gui.server import create_app
from bytestts.gui.session import Session


def panel(session: Session) -> TestClient:
    return api_client(create_app(port=PORT, secret=SECRET, max_threads=2, session=session))


def test_save_dialog_returns_a_ticket_not_a_path(tmp_path: Path) -> None:
    asked: list[tuple[str, str]] = []
    chosen = tmp_path / "x.wav"

    def ask(name: str, file_type: str) -> str:
        asked.append((name, file_type))
        return str(chosen)

    session = Session(SaveTickets(), ask_save_path=ask)
    r = panel(session).post("/api/save-dialog", json={"name": "es-css10 15:02:07", "extension": "wav"})
    assert r.status_code == 200
    assert asked == [("es-css10 15-02-07.wav", "WAV audio (*.wav)")]
    body = r.json()
    assert body["name"] == "x.wav"
    assert str(tmp_path) not in r.text and body["ticket"] != str(chosen)
    assert session.tickets.redeem(body["ticket"]) == chosen


def test_save_dialog_cancel_bad_extension_and_browser_mode(tmp_path: Path) -> None:
    calls: list[str] = []

    def cancel(name: str, file_type: str) -> None:
        calls.append(name)
        return None

    client = panel(Session(SaveTickets(), ask_save_path=cancel))
    r = client.post("/api/save-dialog", json={"name": "a", "extension": "wav"})
    assert r.status_code == 200 and r.json() == {"ticket": None, "name": None}
    r = client.post("/api/save-dialog", json={"name": "a", "extension": "exe"})
    assert r.status_code == 400 and r.json() == {"error": "extension exe is not available"}
    r = client.post("/api/save-dialog", json={"name": "a", "extension": "wav", "path": "C:/x"})
    assert r.status_code == 400
    assert calls == ["a.wav"]

    r = panel(Session(SaveTickets())).post("/api/save-dialog", json={"name": "a", "extension": "wav"})
    assert r.status_code == 400 and r.json() == {"error": "the Save dialog is only in the app window"}


def test_save_dialog_refuses_another_extension(tmp_path: Path) -> None:
    session = Session(SaveTickets(), ask_save_path=lambda name, file_type: str(tmp_path / "x.exe"))
    r = panel(session).post("/api/save-dialog", json={"name": "a", "extension": "wav"})
    assert r.status_code == 400 and r.json() == {"error": "choose a file name ending in .wav"}
    upper = Session(SaveTickets(), ask_save_path=lambda name, file_type: str(tmp_path / "x.WAV"))
    r = panel(upper).post("/api/save-dialog", json={"name": "a", "extension": "wav"})
    assert r.status_code == 200 and r.json()["name"] == "x.WAV"


def test_save_dialog_one_at_a_time(tmp_path: Path) -> None:
    release = threading.Event()
    opened = threading.Event()

    def slow(name: str, file_type: str) -> str:
        opened.set()
        release.wait(10)
        return str(tmp_path / "x.wav")

    client = panel(Session(SaveTickets(), ask_save_path=slow))
    first: list[int] = []
    worker = threading.Thread(
        target=lambda: first.append(client.post("/api/save-dialog", json={"name": "a", "extension": "wav"}).status_code)
    )
    worker.start()
    wait_until(opened.is_set)
    r = client.post("/api/save-dialog", json={"name": "b", "extension": "wav"})
    assert r.status_code == 409 and r.json() == {"error": "the Save dialog is already open"}
    release.set()
    worker.join(10)
    assert first == [200]


def test_unsaved_route_sets_the_session_flag() -> None:
    session = Session(SaveTickets())
    client = panel(session)
    assert client.post("/api/unsaved", json={"unsaved": True}).json() == {"unsaved": True}
    assert session.unsaved is True
    assert client.post("/api/unsaved", json={"unsaved": False}).json() == {"unsaved": False}
    assert session.unsaved is False
    assert client.post("/api/unsaved", json={"unsaved": "yes"}).status_code == 400
    assert session.unsaved is False


def test_status_reports_whether_the_save_dialog_exists() -> None:
    session = Session(SaveTickets())
    client = panel(session)
    assert client.get("/api/status").json()["save_dialog"] is False
    session.ask_save_path = lambda name, file_type: None
    assert client.get("/api/status").json()["save_dialog"] is True


def test_dialog_routes_need_the_secret() -> None:
    session = Session(SaveTickets(), ask_save_path=lambda name, file_type: None)
    app = create_app(port=PORT, secret=SECRET, max_threads=2, session=session)
    bare = TestClient(app, base_url=f"http://127.0.0.1:{PORT}")
    assert bare.post("/api/save-dialog", json={"name": "a", "extension": "wav"}).status_code == 401
    assert bare.post("/api/unsaved", json={"unsaved": True}).status_code == 401
    assert session.unsaved is False
