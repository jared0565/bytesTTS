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


def test_page_explains_a_stopped_panel() -> None:
    """A fetch that cannot reach the server says so in the panel's words, not the browser's "Failed to fetch"."""
    assert '"The panel has stopped: start it again with bytestts gui."' in read("app.js")


def test_page_never_sends_the_user_back_to_the_printed_address() -> None:
    """The printed address works once, and in the browser the launcher has used it already: opening it again would
    stop the panel. A page that lost its secret points to the panel's own tab or window instead."""
    script = read("app.js")
    assert "printed in the terminal" not in script
    assert (
        "\"This page is not connected to the running panel: switch to the panel's own tab or window, or start it"
        ' again with bytestts gui."' in script
    )


def test_replace_question_exists() -> None:
    """A new Speak over an unsaved recording asks first; the question lives in the Speak button's handler."""
    html = read("index.html")
    start = html.index('<dialog id="replace-dialog"')
    dialog = html[start : html.index("</dialog>", start)]
    assert "Replace the unsaved recording?" in dialog
    page = Page()
    page.feed(dialog + "</dialog>")
    assert sorted(c["text"].strip() for c in page.controls if c["tag"] == "button") == ["Keep it", "Replace"]
    script = read("app.js")
    assert '$("speak").addEventListener("click", onSpeakClick)' in script
    assert '$("speak").addEventListener("click", speak)' not in script
    assert '$("replace-dialog").returnValue = ""' in script  # Esc keeps the last choice: clear it before asking


def test_job_stays_running_until_its_audio_is_fetched() -> None:  # Review Focus 3
    """Between a job's last poll and its audio arriving, Speak stays locked: a second press there would start a job
    whose recording the first one's then overwrites."""
    script = read("app.js")
    poll = script[script.index("async function pollJob(") : script.index("\nfunction loadRecording(")]
    fetch = poll.index('api("/jobs/" + id + "/audio"')
    assert "state.fetchingAudio = true;" in poll[:fetch]
    assert poll.rindex("state.fetchingAudio = true;", 0, fetch) > poll.index("state.job = null;\n  if (s.state")
    after = poll[fetch:]
    assert "finally {" in after and "state.fetchingAudio = false;" in after[after.index("finally {") :]
    controls = script[script.index("function updateControls(") : script.index("\nfunction applyStatus(")]
    assert "const running = state.job !== null || state.fetchingAudio;" in controls
    assert "fetchingAudio: false," in script[: script.index("};")]  # declared in the state object


def _function(script: str, name: str) -> str:
    """The source of one top-level function in app.js, up to the next top-level line."""
    start = script.index(f"function {name}(")
    end = script.index("\n}\n", start)
    return script[start : end + 3]


def test_speaker_rule_does_not_use_disabled() -> None:
    """The list is disabled for a voice with one speaker or none, so `disabled` can't mean "send no speaker": with a
    filter leaving one match, the server would get null and speak as the default speaker (voice labels design §5)."""
    script = read("app.js")
    assert '$("speaker").disabled ? null' not in script
    assert "speaker: selectedSpeaker(voice)" in _function(script, "speak")


def test_voice_change_and_refresh_are_separate() -> None:  # Review Focus 1
    """Only the Voice list's own change resets the filters; a refresh (a finished download, a failed job) keeps them
    and the chosen speaker."""
    script = read("app.js")
    assert '$("voice").addEventListener("change", onVoiceChange)' in script
    assert '$("voice").addEventListener("change", showVoice)' not in script
    refresh = _function(script, "refreshVoices")
    assert "showVoice()" in refresh and "onVoiceChange" not in refresh
    change = _function(script, "onVoiceChange")
    assert '$("gender").value = ""' in change and '$("accent").value = ""' in change


def test_new_controls_are_labelled() -> None:
    page = parsed()
    for target in ("gender", "accent", "speaker"):
        assert target in page.label_targets
    names = {c["attrs"].get("id"): c["attrs"].get("aria-label") for c in page.controls}
    assert names["sample"] == "Play a sample of this voice"
    assert names["preview"] == "Preview this speaker"
    html = read("index.html")
    assert 'id="filter-row" aria-label="Speaker filters" hidden' in html
    assert html.index('id="sample"') < html.index('id="download"')  # ▶ Sample comes first after the Voice list
    assert html.index('id="filter-row"') < html.index('id="speaker"')  # the Speaker row is under the filters


def test_info_dialog_shows_the_voice_id() -> None:
    html = read("index.html")
    dialog = html[html.index('<dialog id="info-dialog"') : html.index("</dialog>")]
    first = dialog[dialog.index("<dt>") :]
    assert first.startswith('<dt>Voice id</dt><dd id="info-id"')


def test_speaker_copy_is_exact() -> None:
    script = read("app.js")
    for copy in (
        '"No speaker matches"',
        '"No speaker matches these filters: set one back to Any"',
        '"Speaker is now " + pick + ": the filters hide " + old',
        '"Any"',
        '"Not stated ("',
        '"Featured"',
        '"More speakers"',
        '["Female", "Male"]',
    ):
        assert copy in script, copy


def test_voice_change_clears_a_speaker_note() -> None:
    """"Speaker is now p262: the filters hide p228" is about the voice being left: a voice change clears it."""
    change = _function(read("app.js"), "onVoiceChange")
    assert '$("status").textContent.startsWith("Speaker is now ")' in change
    assert 'setStatus("Ready", "ready")' in change
