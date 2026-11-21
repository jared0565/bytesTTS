# SPDX-License-Identifier: MPL-2.0
"""GET /api/voices/{voice_id}/sample: a voice's, or one speaker's, recorded clip (voice labels design §6.4)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from gui_helpers import CLIP_FILES, CLIPS, CODE, PORT, SECRET, BytesServer, api_client, use_sample_details
from starlette.testclient import TestClient

from bytestts.catalog import details
from bytestts.gui.protection import LaunchCode
from bytestts.gui.saving import SaveTickets
from bytestts.gui.server import create_app
from bytestts.gui.session import Session

ONLY_A_SPEAKER = "the sample request takes only a speaker"


def _app() -> object:
    session = Session(SaveTickets())
    return create_app(port=PORT, secret=SECRET, launch_code=LaunchCode(CODE), max_threads=1, session=session)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[BytesServer]:
    with BytesServer(CLIP_FILES) as s:
        use_sample_details(monkeypatch, tmp_path, s.url_base)
        yield s


def test_sample_needs_the_secret(server: BytesServer) -> None:
    client = TestClient(_app(), base_url=f"http://127.0.0.1:{PORT}")  # no Authorization header
    assert client.get("/api/voices/three/sample").status_code == 401
    assert sum(server.hits.values()) == 0


@pytest.mark.parametrize(
    ("query", "status", "error"),
    [
        ("x/sample", 404, "no sample for voice x"),
        ("bare/sample", 404, "no sample for voice bare"),  # a catalogue voice without details
        ("three/sample?speaker=../x", 404, "speaker ../x is not in voice three"),
        ("three/sample?speaker=c", 404, "no sample for speaker c in voice three"),  # described, no clip
        ("bare/sample?speaker=z", 404, "no sample for speaker z in voice bare"),
        ("tiny/sample?speaker=a", 400, "voice tiny has no speakers"),
        ("three/sample?foo=1", 400, ONLY_A_SPEAKER),
        ("three/sample?speaker=a&speaker=b", 400, ONLY_A_SPEAKER),
        ("three/sample?speaker=a&foo=1", 400, ONLY_A_SPEAKER),
    ],
)
def test_sample_refusals(server: BytesServer, query: str, status: int, error: str) -> None:
    reply = api_client(_app()).get(f"/api/voices/{query}")
    assert (reply.status_code, reply.json()) == (status, {"error": error})
    assert sum(server.hits.values()) == 0


def test_sample_serves_the_pinned_clip(server: BytesServer) -> None:
    reply = api_client(_app()).get("/api/voices/three/sample?speaker=a")
    assert reply.status_code == 200 and reply.content == CLIPS["a"]
    assert reply.headers["content-type"] == "audio/wav"
    assert reply.headers["cache-control"] == "no-store"


def test_sample_without_a_speaker_is_the_default_speakers(server: BytesServer) -> None:
    client = api_client(_app())
    assert client.get("/api/voices/three/sample").content == CLIPS["b"]  # the first featured speaker
    assert client.get("/api/voices/tiny/sample").content == CLIPS["reader"]  # the reader of a voice without speakers


def test_sample_fetch_failure_is_one_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with BytesServer(CLIP_FILES) as s:
        use_sample_details(monkeypatch, tmp_path, s.url_base)
    reply = api_client(_app()).get("/api/voices/three/sample?speaker=a")  # the clip's server has stopped
    assert reply.status_code == 502
    error = reply.json()["error"]
    assert error.startswith("could not fetch the sample: ") and "\n" not in error


def test_a_stalled_clip_says_timed_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The server gives up before the page does, so its reply is what the page shows: the same `timed out` as the
    page's own limit, not requests' wording with the host and port."""
    monkeypatch.setattr(details, "SAMPLE_BUDGET_S", 0.5)  # 25 s in use
    with BytesServer(CLIP_FILES) as s:
        use_sample_details(monkeypatch, tmp_path, s.url_base + "/stall")
        reply = api_client(_app()).get("/api/voices/three/sample?speaker=a")
    assert (reply.status_code, reply.json()["error"]) == (502, "could not fetch the sample: timed out")
