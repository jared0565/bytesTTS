# SPDX-License-Identifier: MPL-2.0
"""The app window (spec §6.7, design B): pywebview on the WebView2 backend, with no bridge object.

Everything pywebview-specific sits behind `open_window`, which the launcher calls and tests replace. The page talks
to Python only over the loopback API. The Save dialog and the "unsaved" flag are server routes, and they use the
hooks this module puts on the Session while the window is open.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bytestts.gui.session import Session, one_line

TITLE = "BytesTTS"
SIZE = (1280, 800)
MIN_SIZE = (720, 600)
CLOSE_QUESTION = "Close and lose unsaved recordings?"
START_TIMEOUT_S = 30.0  # a window that has not loaded by then falls back to the browser
NO_PYWEBVIEW = 'the app window needs pywebview: pip install "bytestts[gui]"'

_WEBVIEW2_FAILED = "WebView2 initialization failed"
_CTRL_C_EVENT, _CTRL_BREAK_EVENT = 0, 1
_WM_SYSCOMMAND, _SC_CLOSE = 0x0112, 0xF060


def _may_close(session: Session, confirm: Callable[[str], bool]) -> bool:
    """Whether the window may close: always when quitting or with nothing unsaved, else the user's answer."""
    if session.quitting or not session.unsaved:
        return True
    return bool(confirm(CLOSE_QUESTION))


def _first_path(result: Any) -> str | None:
    """pywebview's dialog result (a tuple, a list, a string or None) as one path, or None."""
    if not result:
        return None
    if isinstance(result, str):
        return result
    return str(result[0])


class _SaveDialog:
    """The Save dialog, always started in a folder that exists: the last one chosen, else Documents, else home.

    pywebview swaps a folder that does not exist for HOMEPATH, which has no drive letter (Task 2, probe 3).
    """

    def __init__(self, show: Callable[[str, str, str], Any]) -> None:
        self._show = show
        self._last: Path | None = None

    def _start_folder(self) -> Path:
        home = Path.home()
        for folder in (self._last, home / "Documents"):
            if folder is not None and folder.is_dir():
                return folder
        return home

    def __call__(self, file_name: str, file_type: str) -> str | None:
        chosen = _first_path(self._show(str(self._start_folder()), file_name, file_type))
        if chosen is not None:
            self._last = Path(chosen).parent
        return chosen


def _on_console_ctrl(kind: int, close: Callable[[], None]) -> bool:
    """Ctrl+C and Ctrl+Break close the window the way its X does: one question if work is unsaved.

    Answering them here (True) keeps pywebview's own SIGINT handler from running. Its flag is never cleared, so
    after a Cancel it would ask again every half second until the user gave in (Task 2, probe 3).
    """
    if kind in (_CTRL_C_EVENT, _CTRL_BREAK_EVENT):
        close()
        return True
    return False


def _post_sc_close(hwnd: int) -> None:
    if sys.platform == "win32":
        from ctypes import wintypes

        ctypes.windll.user32.PostMessageW(
            wintypes.HWND(hwnd), wintypes.UINT(_WM_SYSCOMMAND), wintypes.WPARAM(_SC_CLOSE), wintypes.LPARAM(0)
        )


def _close_like_x(window: Any, session: Session, *, post: Callable[[int], None] = _post_sc_close) -> None:
    """Post the message a click on X sends. Before the window has a native handle, quit instead."""
    native = getattr(window, "native", None)
    if native is None:
        session.request_quit()
        return
    post(int(native.Handle.ToInt64()))


def _watch_console(handler: Callable[[int], bool]) -> Callable[[], None]:
    """Register `handler` for console control events and return the function that removes it.

    Off Windows, or without a console, there is nothing to register.
    """
    if sys.platform != "win32":
        return lambda: None
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    def answer(kind: int) -> bool:
        try:
            return handler(kind)
        except Exception:  # an error here must not hand Ctrl+C back to pywebview's prompt loop
            return True

    callback = proto(answer)
    if not kernel32.SetConsoleCtrlHandler(callback, True):
        return lambda: None

    def unwatch() -> None:
        kernel32.SetConsoleCtrlHandler(callback, False)  # the closure keeps `callback` alive until then

    return unwatch


class _StartWatch(logging.Handler):
    """Catches pywebview's "WebView2 initialization failed" error.

    pywebview logs that error and carries on with a blank window instead of raising (Task 2, probe 1 (e)).
    """

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.failure: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if self.failure is None and message.startswith(_WEBVIEW2_FAILED):
            first = message.split(" ---> ")[0]  # .NET's inner exception and stack trace follow the arrow
            self.failure = " ".join(first.split())[:300]


def open_window(url: str, session: Session) -> str | None:
    """Show the panel in the app window and block until it closes.

    Returns None after a normal close, having asked the session to quit. Returns a one-line reason when the window
    can't start; the session is left running, so the launcher can open the browser instead.
    """
    try:
        import webview
    except ImportError:
        return NO_PYWEBVIEW

    window = webview.create_window(TITLE, url, js_api=None, width=SIZE[0], height=SIZE[1], min_size=MIN_SIZE)
    if window is None:  # pywebview: only when an `initialized` handler cancels it, and we register none
        return "the app window was not created"
    window.events.closing += lambda: _may_close(
        session, lambda message: window.create_confirmation_dialog(TITLE, message)
    )
    session.close_window = window.destroy
    session.ask_save_path = _SaveDialog(
        lambda folder, name, file_type: window.create_file_dialog(
            webview.FileDialog.SAVE, directory=folder, save_filename=name, file_types=(file_type,)
        )
    )
    watch = _StartWatch()
    failure: list[str] = []

    def watchdog() -> None:
        deadline = time.monotonic() + START_TIMEOUT_S
        while not window.events.loaded.wait(0.1):
            if watch.failure is not None:
                failure.append(f"the app window could not start: {watch.failure}")
                break
            if time.monotonic() >= deadline:
                failure.append(f"the app window did not load within {START_TIMEOUT_S:g} s")
                break
        else:
            return
        window.destroy()

    pywebview_log = logging.getLogger("pywebview")
    pywebview_log.addHandler(watch)
    unwatch = _watch_console(lambda kind: _on_console_ctrl(kind, lambda: _close_like_x(window, session)))
    try:
        webview.start(watchdog, gui="edgechromium", private_mode=True, debug=False)
    except Exception as e:  # pythonnet and .NET start-up errors arrive as many types
        failure.append(f"the app window could not start: {one_line(e)}")
    finally:
        unwatch()
        pywebview_log.removeHandler(watch)
        session.ask_save_path = None
        session.close_window = None
    if failure:
        return failure[0]
    session.request_quit()
    return None
