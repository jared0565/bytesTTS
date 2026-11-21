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
    assert f"function {name}(" in script, f"app.js has no {name}()"
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
    """ "Speaker is now p262: the filters hide p228" is about the voice being left: a voice change clears it."""
    change = _function(read("app.js"), "onVoiceChange")
    assert '$("status").textContent.startsWith("Speaker is now ")' in change
    assert "settleStatus();" in change  # Ready, or what a running task is doing


def _samples(script: str) -> str:
    """The samples section of app.js, between its two markers."""
    marker = "// ---- Samples (voice labels design §5)"
    assert marker in script and "// ---- End of samples" in script, "app.js has no samples section"
    start = script.index(marker)
    return script[start : script.index("// ---- End of samples", start)]


def test_sampler_is_not_the_player() -> None:
    samples = _samples(read("app.js"))
    assert "const sampler = new Audio();" in samples
    assert samples.count('$("player")') == samples.count('$("player").pause()') >= 1  # only ever to pause it


def test_samples_never_touch_the_recording() -> None:
    samples = _samples(read("app.js"))
    for name in ("loadRecording", "setUnsaved", "state.recording"):
        assert name not in samples, name


def test_no_media_src_points_at_the_api() -> None:
    """Recordings and clips come through api() with the secret, as blobs: never from a link to /api."""
    for name in ("app.js", "index.html"):
        text = read(name).replace(" ", "")
        assert 'src="/api' not in text and "src='/api" not in text, name


def test_api_passes_an_abort_signal() -> None:  # Review Focus 2
    """The page's 30 s limit on a clip aborts the fetch: that must not read as "The panel has stopped"."""
    api = _function(read("app.js"), "api")
    assert "signal: options.signal" in api
    caught = api[api.index("} catch (e) {") :]
    assert caught.index('e.name === "AbortError"') < caught.index("SERVER_GONE")


def test_sample_requests_encode_the_speaker() -> None:
    samples = _samples(read("app.js"))
    assert '"/voices/" + encodeURIComponent(voice.id) + "/sample"' in samples
    assert '"?speaker=" + encodeURIComponent(speaker)' in samples


def test_playing_buttons_change_their_names() -> None:
    samples = _samples(read("app.js"))
    for copy in ('"■ Stop"', '"Stop the sample"', '"Stop the preview"', '"Play a sample of this voice"'):
        assert copy in samples, copy
    assert '"Preview this speaker"' in samples


def test_sample_copy_and_limit() -> None:
    samples = _samples(read("app.js"))
    for copy in ('"Fetching the sample"', '"could not fetch the sample: timed out"', "SAMPLE_TIMEOUT_MS = 30000"):
        assert copy in samples, copy


def test_sampler_and_player_stop_each_other() -> None:
    """Starting the sampler pauses #player; #player starting (or a finished Speak) and Stop stop the sampler; the
    sampler's own events feed updateControls, since Stop's state looked only at #player."""
    script = read("app.js")
    wire = _function(script, "wire")
    assert '$("player").addEventListener("play", stopSample)' in wire
    assert "sampler.addEventListener(event, updateControls)" in wire
    assert '$("sample").addEventListener("click", onSampleClick)' in wire
    assert "stopSample();" in _function(script, "stop")
    assert "sampler.paused" in _function(script, "updateControls")


def test_preview_audio_never_reaches_the_recording() -> None:  # Review Focus 4
    """A live Preview's audio goes to the sampler: it must never become the recording or mark it unsaved."""
    script = read("app.js")
    poll = _function(script, "pollJob")
    assert 'if (state.jobKind === "preview") {' in poll, "pollJob has no preview branch"
    start = poll.index('if (state.jobKind === "preview") {')
    branch = poll[start : poll.index("} else {", start)]
    assert "playPreview(blob)" in branch
    assert "loadRecording" not in branch and "setUnsaved" not in branch
    assert "function playPreview(" in _samples(script)


def test_preview_does_not_queue() -> None:
    """A live Preview on a busy CPU says so and stops there: only Speak waits for its turn."""
    script = read("app.js")
    live = _function(script, "previewLive")
    assert "e.status === 409 ? PREVIEW_BUSY" in live and "waitThenSpeak" not in live
    samples = _samples(script)
    assert 'const PREVIEW_BUSY = "Wait for the running task to finish, then press Preview again";' in samples
    assert 'const PREVIEW_FELL_BACK = "Couldn\'t fetch the sample, so speaking it live";' in samples


def test_preview_job_has_its_kind() -> None:
    """Wherever a job starts, its kind is set with it, so pollJob knows where the audio goes."""
    script = read("app.js")
    assert 'state.job = started.id;\n    state.jobKind = "speak";' in _function(script, "speak")
    assert 'state.job = started.id;\n    state.jobKind = "preview";' in _function(script, "previewLive")
    assert '$("preview").addEventListener("click", onPreviewClick)' in _function(script, "wire")


def test_stopping_a_sample_never_leaves_it_fetching() -> None:
    """Stop, or a voice change, between a clip's arrival and its start finds no pending fetch: the status line must
    still leave "Fetching the sample", so that reset can't depend on a fetch being pending."""
    stop = _function(read("app.js"), "stopSample")
    assert '\n  if ($("status").textContent === FETCHING_SAMPLE) {\n    settleStatus();\n  }\n' in stop


def test_status_lines_name_the_voice() -> None:
    """A download's status lines name the voice as the list does, not by its id (`Downloading English · VCTK`)."""
    script = read("app.js")
    start = _function(script, "startDownload")
    assert '"Downloading ") + voice.name' in start and '+ voice.id, "busy")' not in start
    assert 'voiceName(id) + " is ready"' in _function(script, "pollDownload")
    assert "v.id === id" in _function(script, "voiceName") and ".name" in _function(script, "voiceName")


def test_recordings_are_named_by_voice_and_speaker() -> None:
    """A recording, and the file Save writes, is named by the voice's name and the speaker: `English VCTK p253
    06-53-16`. The name's separator becomes a space, since " · " is odd in a file name."""
    script = read("app.js")
    base = _function(script, "recordingBase")
    assert 'voice.name.split(" · ").join(" ")' in base and "speaker" in base
    assert "recordingBase(voice, request.speaker)" in _function(script, "speak")
    assert "recordingName(recordingBase)" in _function(script, "pollJob")


def test_samples_hand_the_status_line_back() -> None:
    """A sample that ends, or is stopped, puts back what a running task says instead of "Ready": a Sample pressed
    during a download used to leave "Ready" while the download went on."""
    script = read("app.js")
    samples = _samples(script)
    assert 'setStatus("Ready"' not in samples and samples.count("settleStatus();") >= 3
    for name in ("onVoiceChange", "updateControls", "stop"):
        assert 'setStatus("Ready"' not in _function(script, name), name
        assert "settleStatus();" in _function(script, name), name


def test_the_settled_status_follows_what_still_runs() -> None:
    """settleStatus() shows a running job's status, else a waiting Speak's, else a download's, else Ready; each
    task records its own line, so a job that ends during a download leaves the download's line, not its own."""
    script = read("app.js")
    settle = _function(script, "settleStatus")
    order = [settle.index(s) for s in ("state.job !== null", "state.waiting", "state.downloading !== null", '"Ready"')]
    assert order == sorted(order)
    assert "state.jobStatus" in settle and "state.downloadStatus" in settle and "WAITING" in settle
    assert "state.downloadStatus = " in _function(script, "startDownload")
    for name in ("speak", "pollJob", "previewLive"):
        assert "setJobStatus(" in _function(script, name), name
    assert "setStatus(WAITING" in _function(script, "waitThenSpeak")


def test_the_voice_actions_wrap_together() -> None:
    """The Voice row's buttons and download progress sit in one group that wraps as a unit: at the window's minimum
    size (i) used to wrap alone onto a second line."""
    html = read("index.html")
    start = html.index('<span class="actions">', html.index('aria-label="Voice"'))
    group = html[start : html.index("</span>\n  </section>", start)]
    order = [
        group.index(f'id="{i}"') for i in ("sample", "download", "repair", "download-bar", "download-text", "cancel")
    ]
    assert order == sorted(order) and group.index('id="info"') > order[-1]
    assert 'id="voice"' not in group
    assert ".actions { display: flex; flex-wrap: wrap;" in read("style.css")
