# SPDX-License-Identifier: MPL-2.0
"""tools/dev/make_samples.py makes the voice sample clips the panel plays, and checks every one (voice labels design
§6.5). Stub voices only: a real-voice run is the operator's."""

import hashlib
import importlib.util
import io
import json
import wave
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from gui_helpers import wait_until
from helpers import TINY

from bytestts.audio import Audio
from bytestts.catalog import catalog, details
from bytestts.catalog.details import Details
from bytestts.catalog.manifest import VoiceManifest
from bytestts.gui import jobs
from bytestts.gui.jobs import SpeakRequest, SpeechJobs
from bytestts.gui.session import CpuSlot

ROOT = Path(__file__).resolve().parents[2]
RATE = 22050
SENTENCE = "Hello. This is how I sound reading your text aloud."


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_samples", ROOT / "tools" / "dev" / "make_samples.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_samples = _load()


def _wav(
    seconds: float = 3.0,
    *,
    amplitude: float = 0.3,
    rate: int = RATE,
    channels: int = 1,
    width: int = 2,
    clipped: float = 0.0,
) -> bytes:
    n = int(seconds * rate)
    x = amplitude * np.sin(2 * np.pi * 440 * np.arange(n) / rate)
    if width == 1:
        data = np.round(x * 127 + 128).astype(np.uint8)
    else:
        data = np.round(x * 32767).astype("<i2")
        k = int(n * clipped)
        data[:k] = np.where(np.arange(k) % 2 == 0, 32767, -32767)
    data = np.repeat(data, channels)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return buf.getvalue()


@pytest.mark.parametrize(
    ("wav", "limit"),
    [
        (_wav(), None),
        (_wav(amplitude=0.0), "1% (-40 dBFS)"),
        (_wav(0.5), "1 to 8 s"),
        (_wav(9.0), "1 to 8 s"),
        (_wav(clipped=0.01), "0.1%"),
        (_wav(channels=2), "mono"),
        (_wav(width=1), "16-bit"),
        (_wav(rate=16000), "22050 Hz"),
        (b"not a wav", "WAV"),
    ],
    ids=["fine", "silent", "short", "long", "clipped", "stereo", "8-bit", "16 kHz", "junk"],
)
def test_check_clip(wav: bytes, limit: str | None) -> None:
    problems = make_samples.check_clip(wav, RATE)
    if limit is None:
        assert problems == []
    else:
        assert len(problems) == 1 and limit in problems[0], problems


class Stub:
    """A voice that speaks two one-second tones per text, silently for the speakers in `silent`."""

    def __init__(self, silent: frozenset[str] = frozenset()) -> None:
        self.silent = silent

    def stream(self, text: str, *, speaker: str | None = None, speed: float = 1.0) -> Iterator[Audio]:
        assert speed == 1.0
        amplitude = 0.0 if speaker in self.silent else 0.3
        t = np.arange(RATE) / RATE
        for tone in (440.0, 660.0):
            yield Audio((amplitude * np.sin(2 * np.pi * tone * t)).astype(np.float32), RATE)


def test_clip_for_matches_the_panel() -> None:
    """The clip is byte for byte what the panel's Speak makes, so a change to the panel's path can't leave the clips
    sounding different from what the user then gets."""
    stub = Stub()
    speech = SpeechJobs(CpuSlot(), on_damaged=lambda voice: None, loader=lambda *a: stub)
    job = speech.start(SpeakRequest(SENTENCE, "x", None, 1.0, "cpu", 1))
    wait_until(lambda: speech.status(job).state != "running")
    assert speech.status(job).state == "done"
    assert make_samples.clip_for(stub, SENTENCE, None) == speech.audio(job)


def _voices(monkeypatch: pytest.MonkeyPatch, stub: Stub) -> list[str]:
    """A catalogue of `two` (speakers a and b) and `one` (a reader); returns the voices loaded, in order."""
    base = VoiceManifest.from_json(TINY / "voice.json")

    def voice(voice_id: str, speakers: list[str]) -> VoiceManifest:
        changes = {"id": voice_id, "speakers": speakers, "featured": [], "sample_rate": RATE, "language": "en"}
        return VoiceManifest(**{**base.__dict__, **changes})

    monkeypatch.setattr(catalog, "load_catalog", lambda: {"two": voice("two", ["a", "b"]), "one": voice("one", [])})
    monkeypatch.setattr(details, "load_details", lambda: Details({"en": SENTENCE}, {}))
    loads: list[str] = []

    def load_voice(voice_id: str, device: str, threads: int) -> Stub:
        assert (device, threads) == ("cpu", 1)
        loads.append(voice_id)
        return stub

    monkeypatch.setattr(jobs, "load_voice", load_voice)
    return loads


def test_main_writes_clips_and_listings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loads = _voices(monkeypatch, Stub())
    out = tmp_path / "out"
    assert make_samples.main([str(out)]) == 0
    assert loads == ["two", "one"]  # one load per voice
    for voice_id, names in {"two": ["a", "b"], "one": ["reader"]}.items():
        listing = json.loads((out / voice_id / "listing.json").read_text(encoding="utf-8"))
        assert [row["name"] for row in listing] == names
        for row in listing:
            data = (out / voice_id / row["file"]).read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            assert row == {
                "voice": voice_id,
                "name": row["name"],
                "file": f"samples/{row['name']}.wav",
                "size": len(data),
                "sha256": sha,
            }
            assert make_samples.check_clip(data, RATE) == []
        assert sorted(p.name for p in (out / voice_id / "samples").iterdir()) == [f"{n}.wav" for n in sorted(names)]
    before = (out / "one" / "listing.json").read_bytes()
    assert make_samples.main([str(out), "--voice", "two"]) == 0
    assert (out / "one" / "listing.json").read_bytes() == before
    assert loads == ["two", "one", "two"]


def test_main_refuses_a_bad_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _voices(monkeypatch, Stub(silent=frozenset({"b"})))
    out = tmp_path / "out"
    (out / "two").mkdir(parents=True)
    (out / "two" / "listing.json").write_text("[]", encoding="utf-8")  # from an earlier run: must not survive
    assert make_samples.main([str(out)]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("make_samples: two/b: ") and "1% (-40 dBFS)" in err[0]
    assert not (out / "two" / "listing.json").exists()
    assert (out / "one" / "listing.json").exists()  # the other voice is fine
    assert make_samples.main([str(out), "--voice", "nope"]) == 1
    assert capsys.readouterr().err.strip().splitlines() == ["make_samples: unknown voice nope"]
