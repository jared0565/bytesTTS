# SPDX-License-Identifier: MPL-2.0
"""The panel server, part 1 (spec §6.3): the page, status, voices and downloads."""

from pathlib import Path

import onnxruntime
import pytest
from gui_helpers import (
    PORT,
    SECRET,
    TINY_SIZE,
    FileServer,
    api_client,
    install,
    tiny_manifest,
    use_catalogue,
    wait_until,
)
from starlette.testclient import TestClient

from bytestts.errors import FrontendUnavailable
from bytestts.gui.protection import CSP
from bytestts.gui.saving import SaveTickets
from bytestts.gui.server import create_app
from bytestts.gui.session import Session
from bytestts.text.phonemizers import espeak


def panel(max_threads: int = 2) -> TestClient:
    return api_client(create_app(port=PORT, secret=SECRET, max_threads=max_threads, session=Session(SaveTickets())))


def test_static_files_and_headers() -> None:
    client = panel()
    page = client.get("/")
    assert page.status_code == 200 and "<title>BytesTTS</title>" in page.text
    assert page.headers["content-type"] == "text/html; charset=utf-8"
    assert page.headers["content-security-policy"] == CSP and page.headers["referrer-policy"] == "no-referrer"
    assert client.get("/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/style.css").headers["content-type"].startswith("text/css")
    for path in ("/etc/passwd", "/static/app.js", "/index.html", "/../pyproject.toml"):
        assert client.get(path).status_code == 404, path


def test_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = panel(max_threads=2)
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    s = client.get("/api/status").json()
    assert (s["max_threads"], s["text_limit"], s["formats"], s["busy"]) == (2, 5000, ["wav"], False)
    assert s["cuda_available"] is False and s["cpu_count"] >= 1 and s["version"]
    monkeypatch.setattr(
        onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert client.get("/api/status").json()["cuda_available"] is True

    def missing() -> None:
        raise FrontendUnavailable("espeak-ng not found")

    monkeypatch.setattr(espeak, "find_espeak", missing)
    found = client.get("/api/status").json()["espeak"]
    assert found == {"found": False, "hint": espeak.INSTALL_HINT}


def test_voices_report_state_and_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    m = tiny_manifest()
    two = tiny_manifest(id="two", speakers=["a", "b"], featured=["b"])
    use_catalogue(monkeypatch, tmp_path, {"tiny": m, "two": two})
    client = panel()
    voices = {v["id"]: v for v in client.get("/api/voices").json()}
    tiny = voices["tiny"]
    assert (tiny["state"], tiny["size_bytes"], tiny["speakers"], tiny["sample_rate"]) == (
        "missing",
        TINY_SIZE,
        [],
        22050,
    )
    assert tiny["licence"] == m.licence and tiny["attribution"] == m.attribution and tiny["language"] == "en"
    assert voices["two"]["speakers"] == ["b", "a"]  # featured first
    install(m)
    assert {v["id"]: v["state"] for v in client.get("/api/voices").json()}["tiny"] == "downloaded"


def test_download_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with FileServer() as server:
        use_catalogue(monkeypatch, tmp_path, {"tiny": tiny_manifest(url_base=server.url_base)})
        client = panel()
        idle = client.get("/api/voices/tiny/download").json()
        assert (idle["state"], idle["done"], idle["total"]) == ("idle", 0, TINY_SIZE)
        started = client.post("/api/voices/tiny/download")
        assert started.status_code == 202 and started.json() == {"state": "running"}
        wait_until(lambda: client.get("/api/voices/tiny/download").json()["state"] != "running")
        done = client.get("/api/voices/tiny/download").json()
        assert done == {"voice": "tiny", "state": "done", "done": TINY_SIZE, "total": TINY_SIZE, "error": None}
        assert client.post("/api/voices/tiny/download/cancel").json() == {"state": "done"}
        assert client.post("/api/voices/nope/download").status_code == 404
        assert client.get("/api/voices/nope/download").status_code == 404


def test_download_while_another_runs_is_409(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with FileServer() as server:
        slow = tiny_manifest(url_base=server.url_base, slow=frozenset({"model.onnx"}))
        other = tiny_manifest(url_base=server.url_base, id="tiny2")
        use_catalogue(monkeypatch, tmp_path, {"tiny": slow, "tiny2": other})
        client = panel()
        assert client.post("/api/voices/tiny/download").status_code == 202
        refused = client.post("/api/voices/tiny2/download")
        assert refused.status_code == 409 and "tiny" in refused.json()["error"]
        server.release.set()
        wait_until(lambda: client.get("/api/voices/tiny/download").json()["state"] != "running")


def test_api_errors_are_one_line_json() -> None:
    client = panel()
    for response in (client.get("/api/nope"), client.post("/api/status")):
        assert response.status_code in (404, 405)
        error = response.json()["error"]
        assert error and "\n" not in error
    no_secret = TestClient(client.app, base_url=f"http://127.0.0.1:{PORT}")
    refused = no_secret.get("/api/status")
    assert refused.status_code == 401 and "\n" not in refused.json()["error"]
