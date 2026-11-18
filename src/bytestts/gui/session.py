# SPDX-License-Identifier: MPL-2.0
"""State shared by the panel's server, its launcher and its window: the session, and the one CPU slot."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from bytestts.gui.saving import SaveTickets


def one_line(error: BaseException) -> str:
    """The CLI's one-line form of an error: its lines joined, never a traceback."""
    return " ".join(part for part in str(error).split() if part) or type(error).__name__


class Busy(Exception):
    """A CPU task (a Speak or an export), or a download, is already running."""


class CpuSlot:
    """At most one CPU task at a time (spec §6.2 rule 9), so the panel keeps at most one core busy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: str | None = None

    def acquire(self, what: str) -> None:
        with self._lock:
            if self._current is not None:
                raise Busy(f"{self._current} is running")
            self._current = what

    def release(self) -> None:
        with self._lock:
            self._current = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None


AskSavePath = Callable[[str, str], "str | None"]
"""Shows the Save dialog for a file name and a file-type filter; returns the chosen path, or None on Cancel."""


@dataclass
class Session:
    """One launch of the panel. The app window sets `close_window` and `ask_save_path` while it is open; in the
    browser both stay None."""

    tickets: SaveTickets
    unsaved: bool = False
    quitting: bool = False
    quit: threading.Event = field(default_factory=threading.Event)
    close_window: Callable[[], None] | None = None
    ask_save_path: AskSavePath | None = None

    def request_quit(self) -> None:
        """Quit: the launcher stops the server when `quit` is set; an open window closes without asking."""
        self.quitting = True
        self.quit.set()
        close = self.close_window
        if close is not None:
            close()
