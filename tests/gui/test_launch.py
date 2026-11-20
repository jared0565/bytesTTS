# SPDX-License-Identifier: MPL-2.0
"""Launching the panel (spec §6.1): the thread cap, the mode, a real loopback server, and `bytestts gui`."""

import json
import logging
import queue
import re
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gui_helpers import install, tiny_manifest, use_catalogue

from bytestts import Audio
from bytestts.cli import main
from bytestts.gui import jobs, launch, window
from bytestts.gui.launch import LaunchOptions, resolve_mode, run_panel, thread_cap
from bytestts.gui.session import Session

URL_RE = re.compile(r"^http://127\.0\.0\.1:(\d+)/\?code=([\w-]{43})$")
REUSED_LINE = (
    "bytestts gui: the launch address was used a second time, so the panel stopped; start it again with bytestts gui"
)
_secrets: dict[str, str] = {}  # per launch address: the secret its one exchange gave


def exchange(url: str) -> tuple[int, Any]:
    """What the page does first: swap the address's launch code for the secret."""
    match = URL_RE.match(url)
    assert match, url
    port, code = match.groups()
    data = json.dumps({"code": code}).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/session", data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - http://127.0.0.1, built above
            reply = json.loads(response.read().decode("utf-8"))
            _secrets[url] = reply["secret"]
            return response.status, reply
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def call(url: str, path: str, method: str = "GET", body: dict[str, Any] | None = None) -> tuple[int, Any]:
    match = URL_RE.match(url)
    assert match, url
    port = match.group(1)
    if url not in _secrets:
        status, reply = exchange(url)
        assert status == 200, reply
    secret = _secrets[url]
    data = json.dumps(body).encode("utf-8") if body is not None else (b"" if method == "POST" else None)
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    request.add_header("Authorization", f"Bearer {secret}")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - http://127.0.0.1, built above
        return response.status, json.loads(response.read().decode("utf-8"))


def start_panel(options: LaunchOptions, *, announce: bool = False) -> tuple[threading.Thread, list[int], str]:
    ready: queue.Queue[str] = queue.Queue()
    result: list[int] = []

    def on_ready(url: str) -> None:
        if announce:
            launch.announce(url)
        ready.put(url)

    thread = threading.Thread(target=lambda: result.append(run_panel(options, on_ready=on_ready)), daemon=True)
    thread.start()
    return thread, result, ready.get(timeout=15)


def test_thread_cap() -> None:
    assert thread_cap(None, {}, 12) == 12
    assert thread_cap(None, {"BYTESTTS_GUI_MAX_THREADS": "2"}, 12) == 2
    assert thread_cap(1, {"BYTESTTS_GUI_MAX_THREADS": "2"}, 12) == 1
    for bad in ("0", "13", "two"):
        with pytest.raises(ValueError, match="BYTESTTS_GUI_MAX_THREADS"):
            thread_cap(None, {"BYTESTTS_GUI_MAX_THREADS": bad}, 12)
    with pytest.raises(ValueError, match="--max-threads"):
        thread_cap(13, {}, 12)


def test_resolve_mode() -> None:
    assert resolve_mode("auto", "win32") == "window"
    assert resolve_mode("auto", "linux") == "browser"
    for mode in ("window", "browser", "none"):
        assert resolve_mode(mode, "linux") == mode  # type: ignore[arg-type]


def test_listen_refuses_ports_outside_0_to_65535() -> None:
    for port in (-1, 65536, -(2**63)):
        with pytest.raises(ValueError, match="port must be from 0 to 65535"):
            launch._listen(port)
    sock = launch._listen(0)
    try:
        assert 1 <= sock.getsockname()[1] <= 65535
    finally:
        sock.close()


def test_serves_on_loopback_until_quit() -> None:
    thread, result, url = start_panel(LaunchOptions(0, "none", 1))
    assert URL_RE.match(url)
    status, body = call(url, "/api/status")
    assert status == 200 and body["max_threads"] == 1 and body["save_dialog"] is False
    assert call(url, "/api/quit", "POST")[0] == 200
    thread.join(10)
    assert not thread.is_alive() and result == [0]


@pytest.mark.skipif(sys.platform != "win32", reason="SO_EXCLUSIVEADDRUSE is Windows-only")
def test_no_other_socket_can_share_the_port() -> None:
    thread, result, url = start_panel(LaunchOptions(0, "none", 1))
    port = int(URL_RE.match(url).group(1))  # type: ignore[union-attr]
    try:
        with socket.socket() as thief:
            thief.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                thief.bind(("127.0.0.1", port))
    finally:
        call(url, "/api/quit", "POST")
        thread.join(10)
    assert result == [0]


def test_window_failure_falls_back_to_browser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """WebView2 fails before it loads the page, so the browser gets the same address with its code still unused."""
    opened: list[str] = []
    exchanged: list[int] = []
    sessions: list[Session] = []

    def no_window(url: str, session: Session) -> str:
        sessions.append(session)
        return "no WebView2"

    def browser(url: str) -> bool:
        opened.append(url)

        def use() -> None:
            try:
                exchanged.append(exchange(url)[0])
            finally:
                sessions[0].request_quit()  # the page's Quit, without depending on the exchange having worked

        threading.Thread(target=use, daemon=True).start()
        return True

    monkeypatch.setattr(window, "open_window", no_window)
    monkeypatch.setattr(webbrowser, "open", browser)
    assert run_panel(LaunchOptions(0, "window", 1), on_ready=lambda url: None) == 0
    assert len(opened) == 1 and URL_RE.match(opened[0])
    assert exchanged == [200]
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1 and "no WebView2" in lines[0] and "browser" in lines[0]


def test_window_close_stops_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(window, "open_window", lambda url, session: calls.append(url))  # returns None at once
    thread, result, url = start_panel(LaunchOptions(0, "window", 1))
    thread.join(10)
    assert not thread.is_alive() and result == [0] and calls == [url]


class BlockedTTS:
    def __init__(self, gate: threading.Event) -> None:
        self.gate = gate

    def stream(self, text: str, *, speaker: str | None = None, speed: float = 1.0) -> Iterator[Audio]:
        self.gate.wait(60)  # "never" set during the test; set afterwards to let the thread go
        yield Audio(np.zeros(10, np.float32), 22050)


def test_quit_while_speaking_returns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gate = threading.Event()
    monkeypatch.setattr(jobs, "load_voice", lambda voice, device, threads: BlockedTTS(gate))
    m = tiny_manifest()
    use_catalogue(monkeypatch, tmp_path, {m.id: m})
    install(m)
    try:
        thread, result, url = start_panel(LaunchOptions(0, "none", 1))
        job = {"text": "Hello.", "voice": m.id, "speed": 1.0, "device": "cpu", "threads": 1}
        assert call(url, "/api/jobs", "POST", job)[0] == 201
        assert call(url, "/api/quit", "POST")[0] == 200
        thread.join(10)
        assert not thread.is_alive() and result == [0]
    finally:
        gate.set()


def test_reuse_while_speaking_returns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gate = threading.Event()
    monkeypatch.setattr(jobs, "load_voice", lambda voice, device, threads: BlockedTTS(gate))
    m = tiny_manifest()
    use_catalogue(monkeypatch, tmp_path, {m.id: m})
    install(m)
    try:
        thread, result, url = start_panel(LaunchOptions(0, "none", 1))
        job = {"text": "Hello.", "voice": m.id, "speed": 1.0, "device": "cpu", "threads": 1}
        assert call(url, "/api/jobs", "POST", job)[0] == 201
        assert exchange(url)[0] == 403
        thread.join(10)
        assert not thread.is_alive() and result == [1]
    finally:
        gate.set()


def test_requests_are_not_logged(capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    """uvicorn's access logger does not propagate to caplog, so the printed output is checked too."""
    caplog.set_level(logging.INFO)
    thread, result, url = start_panel(LaunchOptions(0, "none", 1))
    code = URL_RE.match(url).group(2)  # type: ignore[union-attr]
    # the panel's own http://127.0.0.1 address (URL_RE); its request line carries ?code=<code>
    with urllib.request.urlopen(url, timeout=10) as page:  # noqa: S310
        assert page.status == 200
    call(url, "/api/status")
    secret = _secrets[url]
    call(url, "/api/quit", "POST")
    thread.join(10)
    assert result == [0]
    out, err = capfd.readouterr()
    for value in (code, secret):
        assert value not in out + err
        assert not [r for r in caplog.records if value in r.getMessage()]
    assert "GET /" not in out + err and "POST /" not in out + err
    assert not [r for r in caplog.records if r.name == "uvicorn.access"]


def test_launch_address_never_holds_the_secret(capfd: pytest.CaptureFixture[str]) -> None:
    thread, result, url = start_panel(LaunchOptions(0, "none", 1), announce=True)
    code = URL_RE.match(url).group(2)  # type: ignore[union-attr]
    status, reply = exchange(url)
    assert status == 200
    secret = reply["secret"]
    assert len(secret) == 43 and secret != code and secret not in url
    call(url, "/api/quit", "POST")
    thread.join(10)
    assert result == [0]
    out, err = capfd.readouterr()
    assert f"BytesTTS control panel: {url}" in out
    assert secret not in out + err


def test_reused_launch_address_stops_the_panel(capfd: pytest.CaptureFixture[str]) -> None:
    thread, result, url = start_panel(LaunchOptions(0, "none", 1))
    assert exchange(url)[0] == 200
    status, reply = exchange(url)
    assert status == 403
    assert reply == {
        "error": "this launch address was already used, so the panel has stopped: start it again with bytestts gui"
    }
    thread.join(10)
    assert not thread.is_alive() and result == [1]
    assert capfd.readouterr().err == REUSED_LINE + "\n"


def test_port_in_use_is_exit_1(capsys: pytest.CaptureFixture[str]) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        port = taken.getsockname()[1]
        assert main(["gui", "--no-window", "--port", str(port)]) == 1
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1 and f"127.0.0.1:{port}" in lines[0]


def test_missing_fastapi_is_exit_4(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "fastapi" else real(name, *a))
    assert main(["gui", "--no-window"]) == 4
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1 and "bytestts[gui]" in lines[0]


def test_bad_cap_is_exit_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BYTESTTS_GUI_MAX_THREADS", "0")
    assert main(["gui", "--no-window"]) == 2
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1 and "BYTESTTS_GUI_MAX_THREADS" in lines[0]


def test_bad_port_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["gui", "--no-window", "--port", "70000"]) == 2
    assert "--port" in capsys.readouterr().err
