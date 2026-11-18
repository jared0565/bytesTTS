# SPDX-License-Identifier: MPL-2.0
"""Safety for the control panel tests: no test may open a real browser or a real app window.

A real WebView2 window would appear on the development PC, and on a CI runner it could hang the job. A test
that wants the launcher to "open" something monkeypatches its own stub over these, which wins because the test's
monkeypatch runs after this fixture's.

Nothing to import from here: shared helpers live in tests/gui_helpers.py.
"""

import webbrowser
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _no_real_browser_or_window(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse_browser(*args: Any, **kwargs: Any) -> bool:
        pytest.fail("a test reached the real webbrowser.open")

    monkeypatch.setattr(webbrowser, "open", refuse_browser)
    yield
