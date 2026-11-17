# SPDX-License-Identifier: MPL-2.0
"""The downloader and cache, against a local HTTP server on 127.0.0.1 (no network)."""

import hashlib
import json
import os
import threading
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import requests
from helpers import TINY

from bytestts.catalog import catalog, download
from bytestts.catalog.cache import cache_root, voice_dir
from bytestts.catalog.download import _REPLACE_ATTEMPTS, MAX_REDIRECTS, download_file, ensure_voice
from bytestts.catalog.manifest import FileEntry, VoiceManifest, sha256_file, verify_package
from bytestts.errors import IntegrityError, VoiceNotFound

DATA = bytes(range(256)) * 200
N = len(DATA)
SHA = hashlib.sha256(DATA).hexdigest()
TINY_FILES = {name: (TINY / name).read_bytes() for name in ("model.onnx", "config.json")}


class Server:
    def __init__(self) -> None:
        self.hits: Counter[str] = Counter()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, body: bytes, length: int | None = None) -> None:
                self.send_response(200)
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass  # the client gave up first, as it should

            def _redirect(self, location: str) -> None:
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                path = self.path
                server.hits[path] += 1
                server.encodings.append(self.headers.get("Accept-Encoding", ""))
                if path == "/f":
                    self._send(DATA, N)
                elif path in ("/model.onnx", "/config.json"):
                    body = TINY_FILES[path[1:]]
                    self._send(body, len(body))
                elif path == "/truncated":
                    self._send(DATA[: N // 2], N)  # promises N, closes after N // 2
                elif path == "/huge":
                    self._send(DATA * 10, N * 10)
                elif path == "/huge-unannounced":
                    self._send(DATA * 10)  # no Content-Length: the body runs until the connection closes
                elif path == "/redirect-in":
                    self._redirect("/f")  # relative, resolved against the current URL
                elif path == "/redirect-out":
                    self._redirect("http://example.com/f")
                elif path.startswith("/loop"):
                    self._redirect(f"/loop{server.hits[path]}")
                else:
                    self.send_error(404)

        self.encodings: list[str] = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.handle_error = lambda *a: None  # type: ignore[method-assign]  # aborted bodies are expected
        self.thread = threading.Thread(target=self.httpd.serve_forever, args=(0.05,), daemon=True)  # quick shutdown
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{path}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[Server]:
    s = Server()
    yield s
    s.close()


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every URL the downloader asks requests for."""
    urls: list[str] = []
    real = requests.Session.get

    def spy(self: requests.Session, url: str, **kw: Any) -> Any:
        urls.append(url)
        return real(self, url, **kw)

    monkeypatch.setattr(requests.Session, "get", spy)
    return urls


def leftovers(d: Path) -> list[Path]:
    return sorted(d.iterdir())


# --- download_file ---


def test_downloads_and_verifies(server: Server, tmp_path: Path) -> None:
    assert sha256_file(download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N)) == SHA
    assert leftovers(tmp_path) == [tmp_path / "f"]


def test_asks_for_the_identity_encoding(server: Server, tmp_path: Path) -> None:  # no gzip behind the size cap
    download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N)
    assert server.encodings == ["identity"]


def test_bad_hash_leaves_nothing(server: Server, tmp_path: Path) -> None:
    with pytest.raises(IntegrityError):
        download_file(server.url("f"), tmp_path / "f", sha256="0" * 64, size=N)
    assert leftovers(tmp_path) == []


def test_truncated_response_leaves_nothing(server: Server, tmp_path: Path) -> None:
    with pytest.raises((IntegrityError, requests.RequestException)):
        download_file(server.url("truncated"), tmp_path / "f", sha256=SHA, size=N)
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize("path", ["huge", "huge-unannounced"])
def test_oversized_body_aborted(server: Server, tmp_path: Path, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    read: list[int] = []
    bodies: list[int] = []
    real = requests.Response.iter_content

    def spy(self: requests.Response, *a: Any, **k: Any) -> Any:
        bodies.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(requests.Response, "iter_content", spy)
    with pytest.raises(IntegrityError):
        download_file(server.url(path), tmp_path / "f", sha256=SHA, size=N, progress=lambda done, _: read.append(done))
    if path == "huge":
        assert bodies == []  # refused on its Content-Length, before reading any of the body
    assert max(read, default=0) <= N  # never kept reading past the pinned size
    assert leftovers(tmp_path) == []


def test_http_error_raises_and_leaves_nothing(server: Server, tmp_path: Path) -> None:
    with pytest.raises(requests.HTTPError):
        download_file(server.url("missing"), tmp_path / "f", sha256=SHA, size=N)
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/f",
        "http://localhost.example.com/f",
        "http://127.0.0.1@example.com/f",  # userinfo, not the host
        "http://[::1]/f",
        "ftp://example.com/f",
        "file:///etc/passwd",
        "https:///f",  # no host
    ],
)
def test_refused_urls(tmp_path: Path, url: str, requested: list[str]) -> None:
    with pytest.raises(ValueError):
        download_file(url, tmp_path / "f", sha256=SHA, size=N)
    assert requested == [] and leftovers(tmp_path) == []


def test_https_url_is_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def stop(self: requests.Session, url: str, **kw: Any) -> Any:
        seen.append(url)
        raise requests.ConnectionError("offline test")

    monkeypatch.setattr(requests.Session, "get", stop)
    with pytest.raises(requests.ConnectionError):
        download_file("https://example.com/f", tmp_path / "f", sha256=SHA, size=N)
    assert seen == ["https://example.com/f"] and leftovers(tmp_path) == []


def test_redirect_followed_by_hand(server: Server, tmp_path: Path, requested: list[str]) -> None:
    download_file(server.url("redirect-in"), tmp_path / "f", sha256=SHA, size=N)
    assert requested == [server.url("redirect-in"), server.url("f")]


def test_redirect_to_http_rejected(server: Server, tmp_path: Path, requested: list[str]) -> None:
    with pytest.raises(ValueError):
        download_file(server.url("redirect-out"), tmp_path / "f", sha256=SHA, size=N)
    assert requested == [server.url("redirect-out")]  # rejected before any request leaves the machine
    assert leftovers(tmp_path) == []


def test_too_many_redirects(server: Server, tmp_path: Path, requested: list[str]) -> None:
    with pytest.raises(requests.TooManyRedirects):
        download_file(server.url("loop"), tmp_path / "f", sha256=SHA, size=N)
    assert len(requested) == MAX_REDIRECTS + 1 and leftovers(tmp_path) == []


def test_existing_verified_file_reused(server: Server, tmp_path: Path) -> None:
    download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N)
    download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N)
    assert server.hits["/f"] == 1


def test_existing_bad_file_replaced(server: Server, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"stale")
    download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N)
    assert sha256_file(tmp_path / "f") == SHA


def test_progress_reported(server: Server, tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []
    download_file(
        server.url("f"), tmp_path / "f", sha256=SHA, size=N, progress=lambda done, total: calls.append((done, total))
    )
    assert calls and calls[-1] == (N, N) and all(t == N for _, t in calls)
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)


def test_concurrent_downloads_one_good_file(server: Server, tmp_path: Path) -> None:
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda _: download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N), range(8)))
    assert sha256_file(tmp_path / "f") == SHA and leftovers(tmp_path) == [tmp_path / "f"]


def test_replace_blocked_by_a_verified_file(server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows: os.replace over a file another process holds open raises PermissionError
    def blocked(src: Any, dst: Any) -> None:
        Path(dst).write_bytes(DATA)  # the other process finished first
        raise PermissionError("in use")

    monkeypatch.setattr(download.os, "replace", blocked)
    assert download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N) == tmp_path / "f"
    assert leftovers(tmp_path) == [tmp_path / "f"]


def test_replace_blocked_by_a_bad_file_raises(server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[Any] = []

    def blocked(src: Any, dst: Any) -> None:
        attempts.append(dst)
        raise PermissionError("in use")

    monkeypatch.setattr(download.os, "replace", blocked)
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N)
    assert len(attempts) == _REPLACE_ATTEMPTS and leftovers(tmp_path) == []


def test_replace_retried_while_blocked(server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows: dest is briefly unopenable while another thread replaces it (24 of 240 concurrent downloads failed)
    real, attempts = os.replace, []

    def twice_blocked(src: Any, dst: Any) -> None:
        attempts.append(dst)
        if len(attempts) <= 2:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(download.os, "replace", twice_blocked)
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    assert sha256_file(download_file(server.url("f"), tmp_path / "f", sha256=SHA, size=N)) == SHA
    assert len(attempts) == 3 and leftovers(tmp_path) == [tmp_path / "f"]


# --- cache ---


def test_cache_root_honours_bytestts_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BYTESTTS_HOME", str(tmp_path / "ä cache"))
    assert cache_root() == tmp_path / "ä cache"


def test_cache_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BYTESTTS_HOME", raising=False)
    root = cache_root()
    assert root.is_absolute() and "bytestts" in root.parts[-2:] + root.parts[-3:-2]


def test_voice_dir_is_keyed_by_the_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BYTESTTS_HOME", str(tmp_path))
    m = VoiceManifest.from_json(TINY / "voice.json")
    assert voice_dir(m) == tmp_path / "voices" / "tiny" / m.manifest_digest()[:16]


# --- ensure_voice ---


def tiny_with_urls(server: Server) -> VoiceManifest:
    m = VoiceManifest.from_json(TINY / "voice.json")
    files = {name: FileEntry(e.sha256, e.size, server.url(name)) for name, e in m.files.items()}
    return VoiceManifest(**{**m.__dict__, "files": files})


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("BYTESTTS_HOME", str(tmp_path / "ä cache"))
    return tmp_path / "ä cache"


def test_ensure_voice_downloads_and_verifies(server: Server, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tiny_with_urls(server)
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": m})
    d = ensure_voice("tiny")
    assert d == voice_dir(m) and verify_package(d) == m
    assert ensure_voice("tiny") == d and ensure_voice("tiny", offline=True) == d
    assert server.hits == Counter({"/model.onnx": 1, "/config.json": 1})  # cached after the first call


def test_ensure_voice_repairs_a_tampered_file(server: Server, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": tiny_with_urls(server)})
    d = ensure_voice("tiny")
    (d / "config.json").write_bytes(b"{}")
    assert ensure_voice("tiny") == d and (d / "config.json").read_bytes() == TINY_FILES["config.json"]
    assert server.hits["/config.json"] == 2 and server.hits["/model.onnx"] == 1


def test_ensure_voice_rewrites_an_edited_manifest(server: Server, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the catalogue is the authority: a cached voice.json that differs from it is replaced, not trusted
    m = tiny_with_urls(server)
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": m})
    d = ensure_voice("tiny")
    edited = json.loads((d / "voice.json").read_bytes())
    edited["attribution"] = "someone else"
    (d / "voice.json").write_bytes(json.dumps(edited).encode())
    assert verify_package(ensure_voice("tiny", offline=True)) == m


def test_offline_missing_raises(home: Path, monkeypatch: pytest.MonkeyPatch, requested: list[str]) -> None:
    m = VoiceManifest.from_json(TINY / "voice.json")
    files = {name: FileEntry(e.sha256, e.size, "https://example.com/" + name) for name, e in m.files.items()}
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": VoiceManifest(**{**m.__dict__, "files": files})})
    with pytest.raises(VoiceNotFound, match="offline"):  # a known voice, so the offline branch is what fails
        ensure_voice("tiny", offline=True)
    assert requested == []


def test_unknown_voice_is_not_an_offline_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {})
    with pytest.raises(VoiceNotFound, match="unknown"):
        ensure_voice("nope", offline=True)


def test_catalogue_file_without_url_raises(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": VoiceManifest.from_json(TINY / "voice.json")})
    with pytest.raises(IntegrityError, match="URL"):
        ensure_voice("tiny")


def test_no_part_files_survive_a_failed_voice(server: Server, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tiny_with_urls(server)
    bad = {**m.files, "config.json": FileEntry("0" * 64, m.files["config.json"].size, server.url("config.json"))}
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": VoiceManifest(**{**m.__dict__, "files": bad})})
    with pytest.raises(IntegrityError):
        ensure_voice("tiny")
    assert not [p for p in home.rglob("*") if p.name.endswith(".part")]
    assert os.path.isdir(home)


# --- redirects out of https (the catalogue's own URLs) ---


def _fake_https(monkeypatch: pytest.MonkeyPatch, redirects: dict[str, str]) -> list[str]:
    """requests.Session.get answering each URL in `redirects` with a 302 there; any other URL is 'unreachable'."""
    seen: list[str] = []

    def get(self: requests.Session, url: str, **kw: Any) -> requests.Response:
        seen.append(url)
        if url not in redirects:
            raise requests.ConnectionError("offline test")
        r = requests.Response()
        r.status_code, r.url, r._content_consumed = 302, url, True  # type: ignore[attr-defined]
        r.headers["Location"] = redirects[url]
        return r

    monkeypatch.setattr(requests.Session, "get", get)
    return seen


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1:8080/admin",  # a downgrade into a local service
        "http://localhost/admin",
        "https://127.0.0.1/admin",
        "https://localhost/admin",
        "https://api.localhost/admin",
        # a trailing dot is the same name to the resolver (a fully qualified one), in any case
        "https://localhost./admin",
        "https://api.localhost./admin",
        "https://LocalHost./admin",
        "https://10.0.0.1/admin",
        "https://192.168.1.1/admin",
        "https://169.254.169.254/latest/meta-data",  # cloud metadata
        "https://[::1]/admin",
        "https://[::ffff:127.0.0.1]/admin",
        "https://0.0.0.0/admin",
        # legacy numeric IPv4 spellings: ipaddress refuses them, but the resolver turns each into 127.0.0.1
        "https://2130706433/admin",
        "https://0x7f000001/admin",
        "https://127.1/admin",
        "https://0177.0.0.1/admin",
        "https://0x7f.0.0.1/admin",
        "https://127.0.0.1./admin",
        "https://010.0.0.1/admin",  # 8.0.0.1 to the resolver, but no spelling ipaddress accepts: refused too
    ],
)
def test_https_never_redirects_inward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    seen = _fake_https(monkeypatch, {"https://mirror.example/f": target})
    with pytest.raises(ValueError):
        download_file("https://mirror.example/f", tmp_path / "f", sha256=SHA, size=N)
    assert seen == ["https://mirror.example/f"] and leftovers(tmp_path) == []  # the inward hop was never requested


@pytest.mark.parametrize(
    "target",
    [
        "https://cafe.be/f",
        "https://cafe.be./f",  # a fully qualified public name stays public
        "https://0x.example/f",
        "https://123.example/f",
        "https://93.184.216.34/f",
        "https://9.9.9.9/f",
    ],
)
def test_https_follows_public_names_and_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:  # the numeric-spelling rule must not catch names that only look numeric, nor canonical public IPs
    seen = _fake_https(monkeypatch, {"https://mirror.example/f": target})
    with pytest.raises(requests.ConnectionError):
        download_file("https://mirror.example/f", tmp_path / "f", sha256=SHA, size=N)
    assert seen == ["https://mirror.example/f", target]


def test_https_redirects_to_other_public_hosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_https(
        monkeypatch, {"https://huggingface.co/f": "https://cdn-lfs.hf.co/f", "https://cdn-lfs.hf.co/f": "/g"}
    )
    with pytest.raises(requests.ConnectionError):
        download_file("https://huggingface.co/f", tmp_path / "f", sha256=SHA, size=N)
    assert seen == ["https://huggingface.co/f", "https://cdn-lfs.hf.co/f", "https://cdn-lfs.hf.co/g"]
