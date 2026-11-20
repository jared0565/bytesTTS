# SPDX-License-Identifier: MPL-2.0
"""Errors never hand the page a path (aramid llm/a05, spec §6.7: the chosen path stays under its ticket).

A failed save names only the file. Anything unexpected gives the page a generic one-line error and prints the
detail on the terminal.
"""

import io
import wave
from pathlib import Path

import pytest
from gui_helpers import CODE, PORT, SECRET, install, tiny_manifest, use_catalogue
from starlette.testclient import TestClient

from bytestts.catalog import catalog
from bytestts.gui import saving
from bytestts.gui.protection import LaunchCode
from bytestts.gui.saving import SaveTickets
from bytestts.gui.server import create_app
from bytestts.gui.session import Session

EXPORT = "/api/export?format=wav&quality=standard&ticket="


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 100)
    return buffer.getvalue()


def client_for(session: Session) -> TestClient:
    app = create_app(port=PORT, secret=SECRET, launch_code=LaunchCode(CODE), max_threads=2, session=session)
    return TestClient(
        app,
        base_url=f"http://127.0.0.1:{PORT}",
        headers={"Authorization": f"Bearer {SECRET}"},
        raise_server_exceptions=False,
    )


def test_save_into_a_missing_folder_names_only_the_file(tmp_path: Path) -> None:
    session = Session(SaveTickets())
    ticket = session.tickets.issue(tmp_path / "gone" / "out.wav")
    r = client_for(session).post(EXPORT + ticket, content=wav_bytes())
    assert r.status_code == 500 and r.json()["error"].startswith("could not save out.wav: ")
    assert str(tmp_path) not in r.text and "gone" not in r.text


def test_locked_target_names_only_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def locked(tmp: Path, target: Path) -> None:
        raise PermissionError(13, "Permission denied", str(target))

    monkeypatch.setattr(saving, "replace_into_place", locked)
    session = Session(SaveTickets())
    ticket = session.tickets.issue(tmp_path / "out.wav")
    r = client_for(session).post(EXPORT + ticket, content=wav_bytes())
    assert r.status_code == 500 and r.json() == {"error": "could not save out.wav: Permission denied"}
    assert str(tmp_path) not in r.text
    assert list(tmp_path.iterdir()) == []  # the .part file went too


def test_unexpected_errors_stay_on_the_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> list[object]:
        raise OSError(2, "No such file or directory", "C:/Users/someone/private/catalog.json")

    monkeypatch.setattr(catalog, "list_voices", broken)
    r = client_for(Session(SaveTickets())).get("/api/voices")
    assert r.status_code == 500
    # OSError(2, ...) is built as FileNotFoundError
    assert r.json() == {"error": "the panel's server hit an unexpected error (FileNotFoundError)"}
    assert "private" not in r.text
    assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "GET /api/voices" in err[0] and "private/catalog.json" in err[0]
    # nothing escapes to the server either: uvicorn would print the traceback
    app = create_app(
        port=PORT, secret=SECRET, launch_code=LaunchCode(CODE), max_threads=2, session=Session(SaveTickets())
    )
    strict = TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers={"Authorization": f"Bearer {SECRET}"})
    assert strict.get("/api/voices").status_code == 500  # raise_server_exceptions is on: an escape would raise here


def test_errors_built_from_input_are_one_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    multi = tiny_manifest(id="multi", speakers=["p1"])
    use_catalogue(monkeypatch, tmp_path, {"multi": multi})
    install(multi)
    client = client_for(Session(SaveTickets()))
    job = {"text": "One.", "voice": "multi", "speaker": None, "speed": 1.0, "device": "cpu", "threads": 1}
    replies = [
        client.post("/api/jobs", json={**job, "voice": "a\nb"}),
        client.post("/api/jobs", json={**job, "speaker": "a\nb"}),
        client.post("/api/save-dialog", json={"name": "n", "extension": "x\ny"}),
        client.post("/api/export?format=a%0Ab", content=wav_bytes()),
    ]
    for r in replies:
        assert r.status_code == 400, r.text
        error = r.json()["error"]
        assert "\n" not in error and ("a b" in error or "x y" in error), error
