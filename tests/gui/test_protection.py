# SPDX-License-Identifier: MPL-2.0
"""The control panel's protection layer (spec §6.2): refusal() rules, then the Guard middleware itself."""

import threading

import pytest
from gui_helpers import PORT, SECRET
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from bytestts.gui.protection import CSP, EXCHANGE_PATH, JSON_BODY_CAP, Guard, LaunchCode, Refusal, refusal

RAW = frozenset({"/api/export"})
AUTH = f"Bearer {SECRET}"
BEARER = {"Authorization": AUTH}


def H(**extra: str) -> Headers:
    headers = {"host": f"127.0.0.1:{PORT}"}
    headers.update({k.replace("_", "-"): v for k, v in extra.items()})
    return Headers(headers=headers)


def check(method: str, path: str, headers: Headers) -> Refusal | None:
    return refusal(method, path, headers, port=PORT, secret=SECRET, raw_body_paths=RAW)


def test_wrong_host_is_403() -> None:
    for host in ("evil.example:8765", "127.0.0.1:9999", "127.0.0.1", "localhost"):
        r = check("GET", "/", H(host=host))
        assert r is not None and r.status == 403, host
    assert check("GET", "/", H(host=f"localhost:{PORT}")) is None
    assert check("GET", "/", H()) is None


def test_missing_host_is_403() -> None:
    r = check("GET", "/", Headers(headers={}))
    assert r is not None and r.status == 403


def test_foreign_origin_is_403() -> None:
    for origin in (f"http://127.0.0.1:{PORT + 1}", "http://evil.example", "null", f"https://127.0.0.1:{PORT}"):
        r = check("GET", "/api/status", H(origin=origin, **{"Authorization": AUTH}))
        assert r is not None and r.status == 403, origin
    assert check("GET", "/", H(origin=f"http://localhost:{PORT}")) is None
    assert check("GET", "/api/status", H(origin=f"http://127.0.0.1:{PORT}", Authorization=AUTH)) is None


def test_cross_site_fetch_is_403() -> None:
    for site in ("cross-site", "same-site", "none"):
        r = check("GET", "/api/status", H(sec_fetch_site=site, Authorization=AUTH))
        assert r is not None and r.status == 403, site
    assert check("GET", "/api/status", H(sec_fetch_site="same-origin", Authorization=AUTH)) is None


def test_api_needs_the_bearer_secret() -> None:
    for headers in (
        H(),
        H(Authorization="Bearer wrong"),
        H(Authorization="Bearer "),
        H(Authorization=f"Basic {SECRET}"),
        H(Authorization=f"bearer {SECRET}x"),
        H(cookie=f"s={SECRET}"),
    ):
        r = check("GET", "/api/status", headers)
        assert r is not None and r.status == 401, dict(headers)
    r = check("GET", f"/api/status?s={SECRET}", H())  # a secret in the query never opens the API
    assert r is not None and r.status == 401
    assert check("GET", "/api/status", H(Authorization=f"Bearer {SECRET}")) is None


def test_static_needs_no_secret() -> None:
    assert check("GET", "/", H()) is None
    assert check("GET", "/app.js", H()) is None


def test_json_body_caps() -> None:
    auth = AUTH
    r = check("POST", "/api/jobs", H(content_length="65537", Authorization=auth))
    assert r is not None and r.status == 413
    r = check("POST", "/api/jobs", H(transfer_encoding="chunked", Authorization=auth))
    assert r is not None and r.status == 411
    assert check("POST", "/api/jobs", H(content_length="65536", Authorization=auth)) is None
    assert check("POST", "/api/export", H(content_length="65537", Authorization=auth)) is None  # its own cap
    r = check("POST", "/api/jobs", H(content_length="lots", Authorization=auth))
    assert r is not None and r.status == 400


def test_refusals_are_one_line() -> None:
    r = check("GET", "/api/status", H(host="evil.example"))
    assert r is not None and r.error and "\n" not in r.error


def test_launch_code_redeems_once() -> None:
    code = LaunchCode("c" * 43)
    assert code.redeem("x") == "wrong"
    assert code.redeem("c" * 43) == "ok"
    assert code.redeem("c" * 43) == "used"
    assert code.redeem("x") == "wrong"
    assert code.redeem("") == "wrong"
    with pytest.raises(ValueError):
        LaunchCode("")


def test_launch_code_has_one_winner() -> None:
    code = LaunchCode("c" * 43)
    barrier = threading.Barrier(16)
    results: list[str] = []
    lock = threading.Lock()

    def redeem() -> None:
        barrier.wait(10)
        outcome = code.redeem("c" * 43)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=redeem) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(results) == ["ok"] + ["used"] * 15


def test_exchange_needs_no_secret_but_every_other_check() -> None:
    assert EXCHANGE_PATH == "/api/session"
    assert check("POST", EXCHANGE_PATH, H(content_length="20")) is None
    for headers in (
        H(host="evil.example:8765"),
        H(origin="http://evil.example"),
        H(sec_fetch_site="cross-site"),
    ):
        r = check("POST", EXCHANGE_PATH, headers)
        assert r is not None and r.status == 403, dict(headers)
    r = check("POST", EXCHANGE_PATH, H(content_length=str(JSON_BODY_CAP + 1)))
    assert r is not None and r.status == 413
    r = check("POST", EXCHANGE_PATH, H(transfer_encoding="chunked"))
    assert r is not None and r.status == 411
    for method, path in (("GET", EXCHANGE_PATH), ("GET", "/api/status"), ("POST", EXCHANGE_PATH + "/")):
        r = check(method, path, H())
        assert r is not None and r.status == 401, (method, path)


@pytest.fixture
def guarded() -> TestClient:
    async def page(request: Request) -> PlainTextResponse:
        return PlainTextResponse("page")

    async def api(request: Request) -> PlainTextResponse:
        return PlainTextResponse("api")

    app = Starlette(routes=[Route("/", page), Route("/api/x", api)])
    return TestClient(Guard(app, port=PORT, secret=SECRET, raw_body_paths=RAW), base_url=f"http://127.0.0.1:{PORT}")


def test_guard_adds_headers_and_refuses_as_json(guarded: TestClient) -> None:
    page = guarded.get("/")
    assert page.status_code == 200 and page.text == "page"
    assert page.headers["content-security-policy"] == CSP
    assert CSP == "default-src 'self'; media-src 'self' blob:; frame-ancestors 'none'"
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["referrer-policy"] == "no-referrer"
    api = guarded.get("/api/x", headers=BEARER)
    assert api.status_code == 200 and api.headers["cache-control"] == "no-store"
    assert api.headers["x-content-type-options"] == "nosniff"
    refused = guarded.get("/", headers={"host": "evil.example"})
    assert refused.status_code == 403
    assert refused.headers["content-type"].startswith("application/json")
    error = refused.json()["error"]
    assert error and "\n" not in error
    assert guarded.get("/api/x").status_code == 401
