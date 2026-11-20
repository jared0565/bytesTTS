# SPDX-License-Identifier: MPL-2.0
"""Launching the control panel (spec §6.1): the loopback server first, then the app window, the browser or neither.

The server runs in a daemon thread and the window (pywebview) on the main thread, which then waits for the session
to quit: the window closing, the page's Quit, or Ctrl+C.
"""

from __future__ import annotations

import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import uvicorn

from bytestts.gui import window
from bytestts.gui.saving import SaveTickets
from bytestts.gui.server import create_app
from bytestts.gui.session import Session

Mode = Literal["auto", "window", "browser", "none"]
MAX_THREADS_ENV = "BYTESTTS_GUI_MAX_THREADS"
_START_TIMEOUT_S = 10.0
_STOP_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class LaunchOptions:
    port: int
    mode: Mode
    max_threads: int  # already resolved by thread_cap


def thread_cap(flag: int | None, env: Mapping[str, str], cpu_count: int) -> int:
    """The panel's thread cap: --max-threads, else BYTESTTS_GUI_MAX_THREADS, else the CPU count."""
    if flag is not None:
        if not 1 <= flag <= cpu_count:
            raise ValueError(f"--max-threads must be from 1 to {cpu_count} (the CPU count), not {flag}")
        return flag
    raw = env.get(MAX_THREADS_ENV, "").strip()
    if not raw:
        return cpu_count
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if not 1 <= value <= cpu_count:
        raise ValueError(f"{MAX_THREADS_ENV} must be a whole number from 1 to {cpu_count} (the CPU count), not {raw!r}")
    return value


def resolve_mode(mode: Mode, platform: str) -> Literal["window", "browser", "none"]:
    if mode == "auto":
        return "window" if platform == "win32" else "browser"
    return mode


def announce(url: str) -> None:
    print(f"BytesTTS control panel: {url}", flush=True)


def _listen(port: int) -> socket.socket:
    if not 0 <= port <= 65535:
        raise ValueError("port must be from 0 to 65535")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":  # no other process may bind this port too, even with SO_REUSEADDR
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", port))
    except OSError as e:
        sock.close()
        raise OSError(f"cannot listen on 127.0.0.1:{port}: {e.strerror or e}") from None
    return sock


def run_panel(options: LaunchOptions, *, on_ready: Callable[[str], None] = announce) -> int:
    """Serve the panel until the session quits; returns the exit code (0)."""
    sock = _listen(options.port)
    port = sock.getsockname()[1]
    secret = secrets.token_urlsafe(32)
    session = Session(SaveTickets())
    app = create_app(port=port, secret=secret, max_threads=options.max_threads, session=session)
    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True, name="bytestts-panel")
    thread.start()
    try:
        deadline = time.monotonic() + _START_TIMEOUT_S
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise OSError(f"the panel's server did not start on 127.0.0.1:{port}")
            time.sleep(0.02)
        url = f"http://127.0.0.1:{port}/?s={secret}"
        on_ready(url)
        mode = resolve_mode(options.mode, sys.platform)
        if mode == "window":
            reason = window.open_window(url, session)
            if reason is None:
                session.request_quit()  # the window closed
            else:
                print(f"bytestts gui: {reason}; opening the browser instead", file=sys.stderr)
                mode = "browser"
        if mode == "browser":
            webbrowser.open(url)
        while not session.quit.wait(0.5):
            pass
    finally:
        server.should_exit = True
        thread.join(_STOP_TIMEOUT_S)
        sock.close()
    return 0
