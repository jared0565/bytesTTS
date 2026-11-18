# SPDX-License-Identifier: MPL-2.0
"""Shared helpers for the control panel tests (tests/gui/). Imported as `gui_helpers` (pythonpath = tests)."""

import shutil
import threading
import time
from collections import Counter
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
from helpers import TINY
from starlette.testclient import TestClient

from bytestts.catalog import catalog
from bytestts.catalog.cache import voice_dir
from bytestts.catalog.manifest import FileEntry, VoiceManifest, safe_name

PORT = 8765
SECRET = "s" * 43  # the length of secrets.token_urlsafe(32)
TINY_FILES = ("config.json", "model.onnx")
TINY_SIZE = 599 + 1_112_670  # config.json + model.onnx, from the fixture's voice.json


def tiny_manifest(*, url_base: str | None = None, slow: frozenset[str] = frozenset(), **changes: Any) -> VoiceManifest:
    """The tiny fixture's voice.json as a catalogue entry. With `url_base`, each file gets a URL on the FileServer
    (`/slow/<name>` for the names in `slow`)."""
    m = VoiceManifest.from_json(TINY / "voice.json")
    if url_base is not None:
        files = {
            name: FileEntry(e.sha256, e.size, f"{url_base}/{'slow/' if name in slow else ''}{name}")
            for name, e in m.files.items()
        }
        changes = {"files": files, **changes}
    return VoiceManifest(**{**m.__dict__, **changes})


def use_catalogue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, voices: dict[str, VoiceManifest]) -> None:
    """An empty voice cache under tmp_path, and a catalogue of exactly `voices`."""
    monkeypatch.setenv("BYTESTTS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(catalog, "load_catalog", lambda: dict(voices))


def install(m: VoiceManifest) -> Path:
    """The tiny fixture's files in the cache, as if `m` had been downloaded."""
    d = voice_dir(m)
    d.mkdir(parents=True, exist_ok=True)
    for name in TINY_FILES:
        shutil.copyfile(TINY / name, d / safe_name(name))
    return d


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout} s")
        time.sleep(0.02)


class FileServer:
    """Serves the tiny fixture's files on 127.0.0.1. `/slow/<name>` sends its first 65,536 bytes, then waits for
    `release` before sending the rest. Anything else is 404. `hits` counts requests per path."""

    FIRST = 65_536

    def __init__(self) -> None:
        self.hits: Counter[str] = Counter()
        self.release = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                server.hits[self.path] += 1
                slow = self.path.startswith("/slow/")
                name = self.path.removeprefix("/slow/").lstrip("/")
                if name not in TINY_FILES:
                    self.send_error(404)
                    return
                body = (TINY / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    if slow:
                        self.wfile.write(body[: server.FIRST])
                        self.wfile.flush()
                        server.release.wait(30)
                        self.wfile.write(body[server.FIRST :])
                    else:
                        self.wfile.write(body)
                except OSError:
                    pass  # the client gave up first

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.handle_error = lambda *a: None  # type: ignore[method-assign]  # aborted bodies are expected
        self.thread = threading.Thread(target=self.httpd.serve_forever, args=(0.05,), daemon=True)
        self.thread.start()
        self.url_base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> "FileServer":
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None) -> None:
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def api_client(app: Any) -> TestClient:
    """A test client on the panel's own address, holding the secret."""
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}", headers={"Authorization": f"Bearer {SECRET}"})
