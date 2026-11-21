# SPDX-License-Identifier: MPL-2.0
"""The panel server, part 1 (spec §6.3): the page, status, voices and downloads."""

from pathlib import Path

import onnxruntime
import pytest
from gui_helpers import (
    CLIP_FILES,
    CODE,
    PORT,
    SECRET,
    TINY_SIZE,
    BytesServer,
    FileServer,
    api_client,
    install,
    tiny_manifest,
    use_catalogue,
    use_sample_details,
    wait_until,
)
from starlette.testclient import TestClient

from bytestts.catalog import details, download
from bytestts.errors import FrontendUnavailable
from bytestts.gui.protection import CSP, JSON_BODY_CAP, LaunchCode
from bytestts.gui.saving import SaveTickets
from bytestts.gui.server import create_app
from bytestts.gui.session import Session
from bytestts.text.phonemizers import espeak


def panel(max_threads: int = 2) -> TestClient:
    session = Session(SaveTickets())
    app = create_app(port=PORT, secret=SECRET, launch_code=LaunchCode(CODE), max_threads=max_threads, session=session)
    return api_client(app)


def exchanging_panel() -> tuple[TestClient, Session]:
    """A client with no secret, as the page is before the exchange."""
    session = Session(SaveTickets())
    app = create_app(port=PORT, secret=SECRET, launch_code=LaunchCode(CODE), max_threads=2, session=session)
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}"), session


WRONG_CODE = "wrong launch code: switch to the panel's own tab or window, or start it again with bytestts gui"
REUSED_CODE = "this launch address was already used, so the panel has stopped: start it again with bytestts gui"
STOP_REASON = "the launch address was used a second time, so the panel stopped; start it again with bytestts gui"


def test_exchange_gives_the_secret_once() -> None:
    client, session = exchanging_panel()
    first = client.post("/api/session", json={"code": CODE})
    assert first.status_code == 200 and first.json() == {"secret": SECRET}
    assert first.headers["cache-control"] == "no-store"
    assert not session.quit.is_set() and session.stop_reason is None
    again = client.post("/api/session", json={"code": CODE})
    assert again.status_code == 403 and again.json() == {"error": REUSED_CODE}
    assert session.stop_reason == STOP_REASON
    assert session.quit.is_set()  # the stop runs after the reply is sent


def test_wrong_or_malformed_code_never_stops_the_panel() -> None:
    client, session = exchanging_panel()
    wrong = client.post("/api/session", json={"code": "x"})
    assert wrong.status_code == 401 and wrong.json() == {"error": WRONG_CODE}
    for body in ({"code": 1}, {}, {"code": ""}, {"code": "c" * 44}, {"code": CODE, "x": 1}, [CODE], "x"):
        r = client.post("/api/session", json=body)
        assert r.status_code == 400, body
        assert "\n" not in r.json()["error"]
    r = client.post("/api/session", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    big = b'{"code": "' + b"c" * JSON_BODY_CAP + b'"}'
    r = client.post("/api/session", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert client.get("/api/session").status_code == 401  # only the POST is let through without the secret
    assert not session.quit.is_set() and session.stop_reason is None
    ok = client.post("/api/session", json={"code": CODE})
    assert ok.status_code == 200 and ok.json() == {"secret": SECRET}


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


def test_favicon_is_served_and_linked() -> None:
    client = panel()
    icon = client.get("/favicon.svg")
    assert icon.status_code == 200 and icon.headers["content-type"] == "image/svg+xml"
    assert icon.headers["content-security-policy"] == CSP
    assert "<script" not in icon.text and "href=" not in icon.text and icon.text.lstrip().startswith("<svg")
    assert '<link rel="icon" href="/favicon.svg" type="image/svg+xml">' in client.get("/").text


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


def test_voices_carry_labels_and_filters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with BytesServer(CLIP_FILES) as server:
        use_sample_details(monkeypatch, tmp_path, server.url_base)
        voices = {v["id"]: v for v in panel().get("/api/voices").json()}
        assert sum(server.hits.values()) == 0  # listing the voices fetches no clip
    three, tiny = voices["three"], voices["tiny"]
    assert three["speakers"] == ["b", "a", "c"]  # unchanged: featured first
    assert three["label"] == "English · Test · 3 speakers, 2 accents"
    assert (three["name"], tiny["name"]) == ("English · Test", "English · Reader")
    assert (three["sample_text"], three["has_sample"], three["reader_label"]) == ("Hello.", True, None)
    assert (three["featured"], three["speakers_with_samples"]) == (["b"], ["b", "a"])
    assert three["speaker_info"] == {
        "a": {"label": "a · Female, 30 · Scottish", "gender": "Female", "accent": "Scottish"},
        "b": {"label": "b · Male, 40 · Welsh, Cardiff", "gender": "Male", "accent": "Welsh"},
        "c": {"label": "c · no clip", "gender": None, "accent": None},
    }
    assert (tiny["label"], tiny["has_sample"], tiny["reader_label"]) == (
        "English · Reader · 1 speaker",
        True,
        "Reader · Female",
    )
    assert (tiny["speaker_info"], tiny["featured"], tiny["speakers_with_samples"]) == ({}, [], [])


def test_voice_without_details_falls_back_to_its_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with BytesServer(CLIP_FILES) as server:
        use_sample_details(monkeypatch, tmp_path, server.url_base)
        bare = {v["id"]: v for v in panel().get("/api/voices").json()}["bare"]
    assert (bare["label"], bare["has_sample"], bare["reader_label"], bare["speakers_with_samples"]) == (
        "bare",
        False,
        None,
        [],
    )
    assert bare["speaker_info"] == {"z": {"label": "z", "gender": None, "accent": None}}
    assert bare["name"] == "bare"
    assert bare["sample_text"] == "Hello."  # the language's sentence: a live Preview still works once downloaded


def test_voices_fetch_no_clip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # Review Focus 5
    def no_network(*args: object, **kwargs: object) -> None:
        raise OSError("no network")

    monkeypatch.setattr(download, "download_file", no_network)
    monkeypatch.setattr(details, "download_file", no_network)
    with BytesServer(CLIP_FILES) as server:
        use_sample_details(monkeypatch, tmp_path, server.url_base)
        reply = panel().get("/api/voices")
        assert reply.status_code == 200 and len(reply.json()) == 3
        assert sum(server.hits.values()) == 0
