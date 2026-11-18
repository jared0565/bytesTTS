# SPDX-License-Identifier: MPL-2.0
"""The control panel's protection layer (spec §6.2): who may talk to the panel's loopback server.

The server listens on 127.0.0.1 only, but every web page open in the user's browser can still send requests
there. So each request must name the panel's own Host (no DNS rebinding), carry no foreign Origin, and, for the
API, come from the panel's own origin and hold the per-launch secret in its Authorization header. The secret is
never read from a cookie or the query: a cookie would be sent by pages on other 127.0.0.1 ports, and a media
GET carries no Origin header.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

API_PREFIX = "/api/"
JSON_BODY_CAP = 65_536  # the job body: 5,000 characters even as \uXXXX escapes fit with room to spare
CSP = "default-src 'self'; media-src 'self' blob:; frame-ancestors 'none'"
STATIC_HEADERS: Mapping[str, str] = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
API_HEADERS: Mapping[str, str] = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


@dataclass(frozen=True)
class Refusal:
    status: int
    error: str


def _hosts(port: int) -> frozenset[str]:
    return frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})


def _secret_matches(authorization: str, secret: str) -> bool:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), secret.encode("utf-8"))


def refusal(
    method: str, path: str, headers: Mapping[str, str], *, port: int, secret: str, raw_body_paths: frozenset[str]
) -> Refusal | None:
    """Why this request must be refused, or None. `headers` must be case-insensitive (Starlette's Headers)."""
    hosts = _hosts(port)
    if headers.get("host", "").lower() not in hosts:
        return Refusal(403, "this panel answers only on its own address")
    origin = headers.get("origin")
    if origin is not None and origin.lower() not in {f"http://{h}" for h in hosts}:
        return Refusal(403, "requests from other sites are refused")
    if not path.startswith(API_PREFIX):
        return None
    site = headers.get("sec-fetch-site")
    if site is not None and site != "same-origin":
        return Refusal(403, "requests from other sites are refused")
    if not _secret_matches(headers.get("authorization", ""), secret):
        return Refusal(401, "missing or wrong panel secret: open the address printed in the terminal")
    if method == "POST" and path not in raw_body_paths:
        if "transfer-encoding" in headers:
            return Refusal(411, "a request body needs a Content-Length")
        length = headers.get("content-length")
        if length is not None:
            if not length.isdigit():
                return Refusal(400, "bad Content-Length")
            if int(length) > JSON_BODY_CAP:
                return Refusal(413, f"request body over {JSON_BODY_CAP} bytes")
    return None


class Guard:
    """Pure ASGI middleware: refuses with one-line JSON, and adds the security headers to every response."""

    def __init__(self, app: ASGIApp, *, port: int, secret: str, raw_body_paths: frozenset[str]) -> None:
        self._app = app
        self._port = port
        self._secret = secret
        self._raw_body_paths = raw_body_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope["type"] != "http":  # no websockets here
            await send({"type": "websocket.close", "code": 1008})
            return
        path: str = scope["path"]
        extra = API_HEADERS if path.startswith(API_PREFIX) else STATIC_HEADERS
        refused = refusal(
            scope["method"],
            path,
            Headers(scope=scope),
            port=self._port,
            secret=self._secret,
            raw_body_paths=self._raw_body_paths,
        )
        if refused is not None:
            response = JSONResponse({"error": refused.error}, status_code=refused.status, headers=dict(extra))
            await response(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in extra.items():
                    headers[name] = value
            await send(message)

        await self._app(scope, receive, send_with_headers)
