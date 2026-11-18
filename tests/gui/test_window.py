# SPDX-License-Identifier: MPL-2.0
"""The app window seam (spec §6.7, design B): run against a fake `webview` module, so no window ever opens.

The fake records what open_window asks of pywebview; its `start` plays the GUI loop by calling a test's `run`.
"""

import logging
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import bytestts.gui.window as window_module
from bytestts.gui.saving import SaveTickets
from bytestts.gui.session import Session
from bytestts.gui.window import (
    CLOSE_QUESTION,
    MIN_SIZE,
    SIZE,
    TITLE,
    _close_like_x,
    _first_path,
    _may_close,
    _on_console_ctrl,
    _SaveDialog,
)
from bytestts.gui.window import open_window as real_open_window  # bound before conftest's guard replaces it

URL = "http://127.0.0.1:8765/?s=" + "s" * 43


class FakeEvent:
    def __init__(self) -> None:
        self.handlers: list[Callable[..., Any]] = []
        self._flag = threading.Event()

    def __iadd__(self, handler: Callable[..., Any]) -> "FakeEvent":
        self.handlers.append(handler)
        return self

    def set(self) -> None:
        self._flag.set()

    def is_set(self) -> bool:
        return self._flag.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._flag.wait(timeout)


class FakeWindow:
    def __init__(self, title: str, url: str | None, **kwargs: Any) -> None:
        self.title, self.url, self.kwargs = title, url, kwargs
        self.events = SimpleNamespace(closing=FakeEvent(), loaded=FakeEvent(), closed=FakeEvent())
        self.native: Any = None
        self.destroyed = threading.Event()
        self.confirm_answer = False
        self.confirm_calls: list[tuple[str, str]] = []
        self.dialog_calls: list[dict[str, Any]] = []
        self.dialog_result: Any = None

    def destroy(self) -> None:
        self.destroyed.set()

    def create_confirmation_dialog(self, title: str, message: str) -> bool:
        self.confirm_calls.append((title, message))
        return self.confirm_answer

    def create_file_dialog(
        self,
        dialog_type: int = 10,
        directory: str = "",
        allow_multiple: bool = False,
        save_filename: str = "",
        file_types: tuple[str, ...] = (),
    ) -> Any:
        self.dialog_calls.append(
            {"type": dialog_type, "directory": directory, "save_filename": save_filename, "file_types": file_types}
        )
        return self.dialog_result


Run = Callable[["FakeWebview", Callable[[], None] | None], None]


class FakeWebview(types.ModuleType):
    def __init__(self, run: Run) -> None:
        super().__init__("webview")
        self.FileDialog = SimpleNamespace(OPEN=10, FOLDER=20, SAVE=30)
        self.windows: list[FakeWindow] = []
        self.start_kwargs: dict[str, Any] | None = None
        self._run = run

    def create_window(self, title: str, url: str | None = None, **kwargs: Any) -> FakeWindow:
        window = FakeWindow(title, url, **kwargs)
        self.windows.append(window)
        return window

    def start(self, func: Callable[[], None] | None = None, **kwargs: Any) -> None:
        self.start_kwargs = kwargs
        self._run(self, func)


def install_fake(monkeypatch: pytest.MonkeyPatch, run: Run) -> FakeWebview:
    fake = FakeWebview(run)
    monkeypatch.setitem(sys.modules, "webview", fake)
    return fake


def loads_then_user_closes(fake: FakeWebview, func: Callable[[], None] | None) -> None:
    fake.windows[0].events.loaded.set()
    if func is not None:
        func()


def test_open_window_without_pywebview_gives_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "webview", None)
    session = Session(SaveTickets())
    reason = real_open_window(URL, session)
    assert reason is not None and "bytestts[gui]" in reason and "\n" not in reason
    assert not session.quit.is_set()


def test_window_has_no_bridge_and_starts_webview2_private_without_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(monkeypatch, loads_then_user_closes)
    session = Session(SaveTickets())
    assert real_open_window(URL, session) is None
    (window,) = fake.windows
    assert (window.title, window.url) == (TITLE, URL) and TITLE == "BytesTTS"
    assert window.kwargs["js_api"] is None  # design B: the page gets no Python object at all
    assert (window.kwargs["width"], window.kwargs["height"]) == SIZE == (1280, 800)
    assert window.kwargs["min_size"] == MIN_SIZE
    # debug would turn F5, Ctrl+R, Back and DevTools back on (pywebview edgechromium.py:287)
    assert fake.start_kwargs == {"gui": "edgechromium", "private_mode": True, "debug": False}
    assert session.quitting and session.quit.is_set()
    assert session.ask_save_path is None and session.close_window is None


def test_window_hooks_while_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "Documents").mkdir()
    session = Session(SaveTickets())
    seen: dict[str, Any] = {}

    def run(fake: FakeWebview, func: Callable[[], None] | None) -> None:
        window = fake.windows[0]
        window.events.loaded.set()
        assert func is not None
        func()
        seen["close_is_destroy"] = session.close_window == window.destroy
        window.dialog_result = (str(tmp_path / "Documents" / "a.wav"),)
        assert session.ask_save_path is not None
        seen["asked"] = session.ask_save_path("a.wav", "WAV audio (*.wav)")
        seen["dialog"] = window.dialog_calls[-1]
        (closing,) = window.events.closing.handlers
        seen["close_clean"] = closing()
        session.unsaved = True
        seen["close_unsaved"] = closing()
        seen["confirm"] = list(window.confirm_calls)

    install_fake(monkeypatch, run)
    assert real_open_window(URL, session) is None
    assert seen["close_is_destroy"] is True
    assert seen["asked"] == str(tmp_path / "Documents" / "a.wav")
    assert seen["dialog"] == {
        "type": 30,
        "directory": str(tmp_path / "Documents"),
        "save_filename": "a.wav",
        "file_types": ("WAV audio (*.wav)",),
    }
    assert seen["close_clean"] is True and seen["close_unsaved"] is False
    assert seen["confirm"] == [("BytesTTS", "Close and lose unsaved recordings?")]


def test_closing_handler_asks_only_with_unsaved_work() -> None:
    session = Session(SaveTickets())
    asked: list[str] = []

    def confirm(message: str) -> bool:
        asked.append(message)
        return False

    assert _may_close(session, confirm) is True and asked == []
    session.unsaved = True
    assert _may_close(session, confirm) is False and asked == [CLOSE_QUESTION]
    assert CLOSE_QUESTION == "Close and lose unsaved recordings?"
    session.quitting = True
    assert _may_close(session, confirm) is True and len(asked) == 1


def test_save_dialog_always_starts_in_an_existing_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    answers: list[Any] = [None, (str(out / "a.wav"),), [str(out / "b.wav")], None]
    folders: list[str] = []

    def show(folder: str, name: str, file_type: str) -> Any:
        folders.append(folder)
        return answers.pop(0)

    dialog = _SaveDialog(show)
    assert dialog("a.wav", "WAV audio (*.wav)") is None  # no Documents folder yet: home
    (tmp_path / "Documents").mkdir()
    assert dialog("a.wav", "WAV audio (*.wav)") == str(out / "a.wav")  # Documents
    assert dialog("b.wav", "WAV audio (*.wav)") == str(out / "b.wav")  # the folder chosen last
    out.rmdir()
    assert dialog("c.wav", "WAV audio (*.wav)") is None  # that folder is gone: Documents again
    assert folders == [str(tmp_path), str(tmp_path / "Documents"), str(out), str(tmp_path / "Documents")]


def test_first_path_normalises_dialog_results() -> None:
    for empty in (None, (), [], ""):
        assert _first_path(empty) is None
    assert _first_path(("a.wav",)) == "a.wav"
    assert _first_path(["b.wav", "c.wav"]) == "b.wav"
    assert _first_path("d.wav") == "d.wav"


def test_webview2_failure_gives_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    pywebview_logger = logging.getLogger("pywebview")
    handlers_before = list(pywebview_logger.handlers)
    seen: dict[str, bool] = {}

    def run(fake: FakeWebview, func: Callable[[], None] | None) -> None:
        pywebview_logger.error(
            "WebView2 initialization failed with exception:\n Couldn't find a compatible Webview2 Runtime (no runtime)."
            " ---> System.IO.FileNotFoundException: The system cannot find the file specified.\n   at Microsoft.Web"
        )
        assert func is not None
        func()  # the watchdog sees the logged error and closes the blank window
        seen["destroyed"] = fake.windows[0].destroyed.is_set()

    install_fake(monkeypatch, run)
    session = Session(SaveTickets())
    reason = real_open_window(URL, session)
    assert seen == {"destroyed": True}
    assert reason is not None and "WebView2" in reason and "no runtime" in reason and "\n" not in reason
    assert "FileNotFoundException" not in reason and "Microsoft.Web" not in reason  # .NET's inner detail stays out
    assert not session.quit.is_set() and not session.quitting  # the launcher falls back to the browser
    assert session.ask_save_path is None and session.close_window is None
    assert pywebview_logger.handlers == handlers_before


def test_window_that_never_loads_gives_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(window_module, "START_TIMEOUT_S", 0.2)
    seen: dict[str, bool] = {}

    def run(fake: FakeWebview, func: Callable[[], None] | None) -> None:
        assert func is not None
        func()
        seen["destroyed"] = fake.windows[0].destroyed.is_set()

    install_fake(monkeypatch, run)
    session = Session(SaveTickets())
    reason = real_open_window(URL, session)
    assert seen == {"destroyed": True}
    assert reason is not None and "did not load" in reason and "\n" not in reason
    assert not session.quit.is_set()


def test_window_not_created_gives_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(monkeypatch, loads_then_user_closes)
    monkeypatch.setattr(fake, "create_window", lambda *args, **kwargs: None)  # pywebview's `Window | None`
    session = Session(SaveTickets())
    reason = real_open_window(URL, session)
    assert reason is not None and "not created" in reason and "\n" not in reason
    assert fake.start_kwargs is None and not session.quit.is_set()
    assert session.ask_save_path is None and session.close_window is None


def test_start_failure_gives_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(fake: FakeWebview, func: Callable[[], None] | None) -> None:
        raise RuntimeError("no .NET\nruntime")

    install_fake(monkeypatch, run)
    session = Session(SaveTickets())
    reason = real_open_window(URL, session)
    assert reason is not None and "no .NET runtime" in reason and "\n" not in reason
    assert not session.quit.is_set()
    assert session.ask_save_path is None and session.close_window is None


def test_open_window_answers_ctrl_c_itself_while_open(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    seen: dict[str, Any] = {}

    def watch(handler: Callable[[int], bool]) -> Callable[[], None]:
        events.append("watch")
        seen["handler"] = handler
        return lambda: events.append("unwatch")

    monkeypatch.setattr(window_module, "_watch_console", watch)

    def run(fake: FakeWebview, func: Callable[[], None] | None) -> None:
        loads_then_user_closes(fake, func)
        events.append("running")
        seen["answered"] = seen["handler"](0)  # Ctrl+C, before the window has a native handle

    install_fake(monkeypatch, run)
    session = Session(SaveTickets())
    assert real_open_window(URL, session) is None
    assert events == ["watch", "running", "unwatch"]
    assert seen["answered"] is True and session.quit.is_set()


def test_ctrl_c_and_break_close_through_the_x_path() -> None:
    calls: list[str] = []
    assert _on_console_ctrl(0, lambda: calls.append("close")) is True  # CTRL_C_EVENT
    assert _on_console_ctrl(1, lambda: calls.append("close")) is True  # CTRL_BREAK_EVENT
    assert _on_console_ctrl(2, lambda: calls.append("close")) is False  # CTRL_CLOSE_EVENT: Windows ends us
    assert calls == ["close", "close"]


def test_close_like_x_posts_sc_close_or_quits() -> None:
    posted: list[int] = []
    session = Session(SaveTickets())
    window = SimpleNamespace(native=SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: 1234)))
    _close_like_x(window, session, post=posted.append)
    assert posted == [1234] and not session.quit.is_set()
    _close_like_x(SimpleNamespace(native=None), session, post=posted.append)
    assert posted == [1234] and session.quit.is_set()


def test_guard_blocks_the_real_window() -> None:
    with pytest.raises(pytest.fail.Exception):
        window_module.open_window(URL, Session(SaveTickets()))
