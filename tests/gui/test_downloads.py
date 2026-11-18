# SPDX-License-Identifier: MPL-2.0
"""Voice downloads from the panel: progress across files, one at a time, Cancel and Repair (spec §6.5)."""

import threading
from pathlib import Path

import pytest
from gui_helpers import TINY_SIZE, FileServer, install, tiny_manifest, use_catalogue, wait_until

from bytestts.catalog.cache import voice_dir
from bytestts.errors import VoiceNotFound
from bytestts.gui.downloads import DownloadCancelled, Downloads, _Progress
from bytestts.gui.session import Busy


def finished(downloads: Downloads, voice: str) -> bool:
    return downloads.status(voice).state != "running"


def test_progress_sums_the_per_file_reports() -> None:
    """ensure_voice reports (done, size) for one file at a time; the panel shows the sum across files."""
    cancel = threading.Event()
    progress = _Progress(1000, cancel)
    assert progress.value == 0
    progress(10, 50)
    assert progress.value == 10
    progress(20, 50)  # the same file again: not finished yet, so nothing is added
    assert progress.value == 20
    progress(50, 50)
    assert progress.value == 50
    progress(5, 900)  # a new file starts after one that finished
    assert progress.value == 55
    progress(900, 900)
    assert progress.value == 950
    cancel.set()
    with pytest.raises(DownloadCancelled):
        progress(0, 10)


def test_state_follows_the_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    m = tiny_manifest()
    use_catalogue(monkeypatch, tmp_path, {"tiny": m})
    downloads = Downloads()
    assert downloads.state("tiny") == "missing"
    d = install(m)
    assert downloads.state("tiny") == "downloaded"
    model = d / "model.onnx"
    model.write_bytes(model.read_bytes()[:-1])
    assert downloads.state("tiny") == "missing"
    install(m)
    downloads.mark_damaged("tiny")
    assert downloads.state("tiny") == "needs-repair"
    with pytest.raises(VoiceNotFound):
        downloads.state("nope")


def test_status_before_any_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_catalogue(monkeypatch, tmp_path, {"tiny": tiny_manifest()})
    s = Downloads().status("tiny")
    assert (s.voice, s.state, s.done, s.total, s.error) == ("tiny", "idle", 0, TINY_SIZE, None)


def test_download_reports_bytes_then_done(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with FileServer() as server:
        use_catalogue(monkeypatch, tmp_path, {"tiny": tiny_manifest(url_base=server.url_base)})
        downloads = Downloads()
        downloads.start("tiny")
        wait_until(lambda: finished(downloads, "tiny"))
    s = downloads.status("tiny")
    assert (s.state, s.done, s.total, s.error) == ("done", TINY_SIZE, TINY_SIZE, None)
    assert downloads.state("tiny") == "downloaded"


def test_second_download_is_busy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with FileServer() as server:
        slow = tiny_manifest(url_base=server.url_base, slow=frozenset({"model.onnx"}))
        other = tiny_manifest(url_base=server.url_base, id="tiny2")
        use_catalogue(monkeypatch, tmp_path, {"tiny": slow, "tiny2": other})
        downloads = Downloads()
        downloads.start("tiny")
        wait_until(lambda: downloads.status("tiny").done > 599)
        with pytest.raises(Busy, match="tiny"):
            downloads.start("tiny2")
        server.release.set()
        wait_until(lambda: finished(downloads, "tiny"))
        assert downloads.status("tiny").state == "done"
        downloads.start("tiny2")  # free again
        wait_until(lambda: finished(downloads, "tiny2"))
        assert downloads.status("tiny2").state == "done"


def test_cancel_removes_the_partial_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with FileServer() as server:
        m = tiny_manifest(url_base=server.url_base, slow=frozenset({"model.onnx"}))
        use_catalogue(monkeypatch, tmp_path, {"tiny": m})
        downloads = Downloads()
        downloads.start("tiny")
        wait_until(lambda: downloads.status("tiny").done > 599)  # config.json done, model.onnx under way
        downloads.cancel("tiny")
        server.release.set()
        wait_until(lambda: finished(downloads, "tiny"))
    assert downloads.status("tiny").state == "cancelled"
    d = voice_dir(m)
    assert not list(d.glob("*.part"))
    assert not (d / "model.onnx").exists()
    assert downloads.state("tiny") == "missing"


def test_repair_refetches_only_the_damaged_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with FileServer() as server:
        m = tiny_manifest(url_base=server.url_base)
        use_catalogue(monkeypatch, tmp_path, {"tiny": m})
        d = install(m)
        data = bytearray((d / "model.onnx").read_bytes())
        data[300] ^= 1  # same size, wrong hash
        (d / "model.onnx").write_bytes(bytes(data))
        downloads = Downloads()
        downloads.mark_damaged("tiny")
        assert downloads.state("tiny") == "needs-repair"
        downloads.start("tiny")
        wait_until(lambda: finished(downloads, "tiny"))
        assert downloads.status("tiny").state == "done"
        assert server.hits["/model.onnx"] == 1 and server.hits["/config.json"] == 0
    assert downloads.state("tiny") == "downloaded"


def test_failed_download_is_one_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with FileServer() as server:
        m = tiny_manifest(url_base=server.url_base + "/missing")
        use_catalogue(monkeypatch, tmp_path, {"tiny": m})
        downloads = Downloads()
        downloads.start("tiny")
        wait_until(lambda: finished(downloads, "tiny"))
    s = downloads.status("tiny")
    assert s.state == "failed" and s.error
    assert "\n" not in s.error and "Traceback" not in s.error


def test_unknown_voice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_catalogue(monkeypatch, tmp_path, {})
    downloads = Downloads()
    with pytest.raises(VoiceNotFound):
        downloads.start("nope")
    with pytest.raises(VoiceNotFound):
        downloads.status("nope")
    downloads.cancel("nope")  # nothing to cancel: no error
