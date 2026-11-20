# SPDX-License-Identifier: MPL-2.0
"""The page's static files (spec §6.2 rule 7, §6.7 design B): no HTML sinks, no inline script or style, nothing
loaded from elsewhere, every control labelled, and no use of a pywebview bridge."""

from html.parser import HTMLParser
from importlib import resources
from typing import Any

FILES = ("index.html", "app.js", "style.css")


def read(name: str) -> str:
    return resources.files("bytestts.gui").joinpath("static", name).read_text(encoding="utf-8")


class Page(HTMLParser):
    """Collects what the tests check: controls with their text, label targets, scripts and inline styles."""

    def __init__(self) -> None:
        super().__init__()
        self.controls: list[dict[str, Any]] = []
        self.label_targets: set[str] = set()
        self.inline_scripts = 0
        self.style_elements = 0
        self.event_attributes: list[tuple[str, str]] = []
        self.style_attributes: list[str] = []
        self._buttons: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        self.event_attributes += [(tag, k) for k in a if k.startswith("on")]
        if "style" in a:
            self.style_attributes.append(tag)
        if tag == "script" and not a.get("src"):
            self.inline_scripts += 1
        if tag == "style":
            self.style_elements += 1
        if tag == "label" and a.get("for"):
            self.label_targets.add(a["for"])
        if tag in ("input", "select", "textarea", "button"):
            control = {"tag": tag, "attrs": a, "text": ""}
            self.controls.append(control)
            if tag == "button":
                self._buttons.append(control)

    def handle_endtag(self, tag: str) -> None:
        if tag == "button" and self._buttons:
            self._buttons.pop()

    def handle_data(self, data: str) -> None:
        for button in self._buttons:
            button["text"] += data


def parsed() -> Page:
    page = Page()
    page.feed(read("index.html"))
    return page


def test_page_has_no_html_injection_sinks() -> None:
    for name in FILES:
        text = read(name)
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            assert sink not in text, f"{name} uses {sink}"


def test_page_has_no_inline_script_or_style() -> None:
    page = parsed()
    assert page.inline_scripts == 0 and page.style_elements == 0
    assert page.event_attributes == [] and page.style_attributes == []


def test_page_loads_nothing_external() -> None:
    for name in FILES:
        text = read(name)
        assert "http://" not in text and "https://" not in text, name


def test_every_control_has_a_label() -> None:
    page = parsed()
    assert len(page.controls) >= 12
    for control in page.controls:
        a = control["attrs"]
        labelled = a.get("id") in page.label_targets or a.get("aria-label") or control["text"].strip()
        assert labelled, f"unlabelled <{control['tag']}> {a}"


def test_page_uses_no_window_bridge() -> None:
    """Design B: the window has no js_api, so the page must not look for one (Save and unsaved are routes)."""
    script = read("app.js")
    assert "pywebview" not in script
    assert "/save-dialog" in script and "/unsaved" in script


def test_page_gets_the_secret_only_from_the_exchange() -> None:
    script = read("app.js")
    assert '"/api/session"' in script and 'get("code")' in script
    assert 'get("s")' not in script
