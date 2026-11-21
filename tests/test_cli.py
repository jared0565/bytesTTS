# SPDX-License-Identifier: MPL-2.0
import io
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from helpers import TEST_THREADS, TINY, edit_json, repin

import bytestts
from bytestts import Audio, BytesTTS
from bytestts.catalog import catalog
from bytestts.catalog.manifest import VoiceManifest
from bytestts.cli import main
from bytestts.errors import FrontendUnavailable

TINY_SR = 22050
T = ["--threads", str(TEST_THREADS)]  # every session in tests is single-threaded (conftest enforces it)


def say(*args: str) -> int:
    return main(["say", *args, *T])


def tampered(tmp_path: Path) -> Path:
    d = tmp_path / "tampered"
    shutil.copytree(TINY, d)
    data = bytearray((d / "model.onnx").read_bytes())
    data[300] ^= 1
    (d / "model.onnx").write_bytes(bytes(data))
    return d


def fake_synthesize(seen: list[str]) -> Any:
    def synthesize(self: BytesTTS, text: str, **kw: Any) -> Audio:
        seen.append(text)
        return Audio(np.zeros(1, np.float32), TINY_SR)

    return synthesize


def one_line_error(err: str) -> bool:
    return "Traceback" not in err and len(err.strip().splitlines()) == 1


# --- say ---


def test_say_to_file(tmp_path: Path) -> None:
    out = tmp_path / "o ü.wav"
    assert say("hi.", "--voice-path", str(TINY), "-o", str(out)) == 0 and out.stat().st_size > 44
    assert out.read_bytes()[:4] == b"RIFF"


def test_say_from_file_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "in.txt"
    f.write_bytes("﻿naïve café.".encode())  # with a BOM, as Notepad writes
    seen: list[str] = []
    monkeypatch.setattr(BytesTTS, "synthesize", fake_synthesize(seen))
    assert say("--file", str(f), "--voice-path", str(TINY), "-o", str(tmp_path / "o.wav")) == 0
    assert seen == ["naïve café."]


def test_say_from_stdin_utf8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # bytes, not the code page
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO("﻿café naïve.".encode()), encoding="cp1252"))
    seen: list[str] = []
    monkeypatch.setattr(BytesTTS, "synthesize", fake_synthesize(seen))
    assert say("-", "--voice-path", str(TINY), "-o", str(tmp_path / "o.wav")) == 0
    assert seen == ["café naïve."]  # reading sys.stdin would give "ï»¿cafÃ© naÃ¯ve." and fail here


def test_stdout_wav(capfdbinary: pytest.CaptureFixture[bytes]) -> None:
    assert say("hi.", "--voice-path", str(TINY), "-o", "-") == 0
    assert capfdbinary.readouterr().out[:4] == b"RIFF"


def test_speaker_and_speed_reach_the_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def synthesize(self: BytesTTS, text: str, **kw: Any) -> Audio:
        seen.update(kw)
        return Audio(np.zeros(1, np.float32), TINY_SR)

    monkeypatch.setattr(BytesTTS, "synthesize", synthesize)
    assert (
        say("hi.", "--voice-path", str(TINY), "--speaker", "p1", "--speed", "1.5", "-o", str(tmp_path / "o.wav")) == 0
    )
    assert seen == {"speaker": "p1", "speed": 1.5}


@pytest.mark.parametrize(("flags", "threads"), [(["--threads", "3"], 3), ([], None)])
def test_device_and_threads_reach_the_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], threads: int | None
) -> None:
    seen: dict[str, Any] = {}
    real = BytesTTS.load_path

    def load_path(path: Any, **kw: Any) -> BytesTTS:
        seen.update(kw)  # what the CLI asked for; the session itself stays single-threaded (conftest)
        return real(path, **{**kw, "threads": TEST_THREADS})

    monkeypatch.setattr(BytesTTS, "load_path", load_path)
    args = ["say", "hi.", "--voice-path", str(TINY), "--device", "cpu", *flags, "-o", str(tmp_path / "o.wav")]
    assert main(args) == 0
    assert seen == {"device": "cpu", "threads": threads}


def test_text_and_file_together_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "in.txt"
    f.write_bytes(b"hello.")  # readable: only the both-given rule can refuse this
    assert say("hi.", "--file", str(f), "--voice-path", str(TINY), "-o", str(tmp_path / "o.wav")) == 2
    assert one_line_error(capsys.readouterr().err)


def test_voice_id_goes_through_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def load(voice_id: str, **kw: Any) -> BytesTTS:
        seen.append((voice_id, kw))
        return BytesTTS.load_path(TINY, **kw)

    monkeypatch.setattr(BytesTTS, "load", load)
    assert say("hi.", "--voice", "tiny", "-o", str(tmp_path / "o.wav")) == 0
    assert seen == [("tiny", {"device": "auto", "threads": TEST_THREADS})]


# --- exit codes ---


@pytest.mark.parametrize(
    "args",
    [
        ["  "],  # NoSpeakableText
        ["hi.", "--threads", "0"],  # 0 would mean all cores
        ["hi.", "--threads", "-2"],
        ["hi.", "--threads", "two"],
        ["hi.", "--speed", "2.5"],
        ["hi.", "--speed", "nan"],
        ["hi.", "--device", "tpu"],
        [],  # no text
    ],
)
def test_usage_errors_exit_2(args: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["say", *args, "--voice-path", str(TINY), "-o", str(tmp_path / "x.wav"), *T])
    assert code == 2 and "Traceback" not in capsys.readouterr().err


def test_voice_and_voice_path_are_exclusive(tmp_path: Path) -> None:
    assert say("hi.", "--voice", "a", "--voice-path", str(TINY), "-o", str(tmp_path / "x.wav")) == 2


def test_unreadable_or_non_utf8_file_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"caf\xe9")  # cp1252, not UTF-8
    for f in (bad, tmp_path / "missing.txt"):
        assert say("--file", str(f), "--voice-path", str(TINY), "-o", str(tmp_path / "o.wav")) == 2
        assert one_line_error(capsys.readouterr().err)


def test_tampered_voice_exit_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert say("hi", "--voice-path", str(tampered(tmp_path)), "-o", str(tmp_path / "o.wav")) == 3
    assert one_line_error(capsys.readouterr().err)


def test_unknown_voice_exit_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {})
    assert say("hi", "--voice", "nope", "-o", str(tmp_path / "o.wav")) == 3


def test_frontend_unavailable_exit_4_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable(*a: Any, **k: Any) -> BytesTTS:
        raise FrontendUnavailable("espeak-ng was not found.\nInstall it from\n  https://example.org")

    monkeypatch.setattr(BytesTTS, "load_path", unavailable)
    assert say("hi", "--voice-path", str(TINY), "-o", str(tmp_path / "o.wav")) == 4
    err = capsys.readouterr().err
    assert one_line_error(err) and "https://example.org" in err


def test_unsupported_device_exit_4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert say("hi", "--voice-path", str(TINY), "--device", "cuda", "-o", str(tmp_path / "o.wav")) == 4


def test_unexpected_error_exit_1_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*a: Any, **k: Any) -> BytesTTS:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(BytesTTS, "load_path", broken)
    assert say("hi", "--voice-path", str(TINY), "-o", str(tmp_path / "o.wav")) == 1
    err = capsys.readouterr().err
    assert one_line_error(err) and "disk on fire" in err


def test_unwritable_output_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert say("hi.", "--voice-path", str(TINY), "-o", str(tmp_path / "no such dir" / "o.wav")) == 1
    assert one_line_error(capsys.readouterr().err)


# --- voices, download, version ---


def _tiny(**kw: Any) -> VoiceManifest:
    return VoiceManifest(**{**VoiceManifest.from_json(TINY / "voice.json").__dict__, **kw})


def test_voices_lists_the_catalogue(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    voices = {"tiny": _tiny(), "de-tiny": _tiny(id="de-tiny", language="de")}
    monkeypatch.setattr(catalog, "load_catalog", lambda: voices)
    assert main(["voices"]) == 0
    out = capsys.readouterr().out
    assert "tiny" in out and "de-tiny" in out
    assert main(["voices", "--language", "de"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("de-tiny")


@pytest.mark.parametrize(
    ("speakers", "shown"), [([], "1 speaker"), (["tux"], "1 speaker"), (["p1", "p2"], "2 speakers")]
)
def test_voices_counts_speakers(
    speakers: list[str], shown: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # a one-row speaker table (the CSS10 voices) is one speaker, as is none
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": _tiny(speakers=speakers)})
    assert main(["voices"]) == 0
    assert capsys.readouterr().out.split("\t")[2] == shown


def test_speakers_lists_the_shortlist_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # the panel's order (server.py _voice_json), so line 1 is the default speaker
    voice = _tiny(speakers=["a", "b", "c"], featured=["c", "a"])
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": voice})
    assert main(["speakers", "tiny"]) == 0
    assert capsys.readouterr().out.splitlines() == ["c\tfeatured\t", "a\tfeatured\t", "b\t\t"]


def test_speakers_of_a_single_speaker_voice_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": _tiny(speakers=[])})
    assert main(["speakers", "tiny"]) == 0
    assert capsys.readouterr().out == ""


def test_speakers_unknown_voice_is_exit_3(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": _tiny()})
    assert main(["speakers", "nope"]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.strip().splitlines() == ["bytestts: error: unknown voice nope"]


def test_speakers_takes_one_voice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": _tiny()})
    assert main(["speakers"]) == 2
    assert main(["speakers", "tiny", "--voice-path", str(tmp_path)]) == 2


def test_speakers_reads_a_package(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    package = tmp_path / "pkg"
    shutil.copytree(TINY, package)
    (package / "speakers.json").write_bytes(b'{"a": 0, "b": 1}')
    repin(package, "speakers.json")
    edit_json(package / "voice.json", lambda m: m.update(speakers=["a", "b"], featured=["b"]))
    assert main(["speakers", "--voice-path", str(package)]) == 0
    assert capsys.readouterr().out.splitlines() == ["b\tfeatured\t", "a\t\t"]
    (package / "speakers.json").write_bytes(b'{"a": 1, "b": 0}')  # edited after it was pinned
    assert main(["speakers", "--voice-path", str(package)]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.strip().splitlines()) == 1


def test_speakers_needs_no_download(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def no(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("bytestts speakers must not download or load a voice")

    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": _tiny(speakers=["a"])})
    monkeypatch.setattr("bytestts.cli.ensure_voice", no)
    monkeypatch.setattr("bytestts.catalog.download.ensure_voice", no)
    monkeypatch.setattr(BytesTTS, "load", no)
    monkeypatch.setattr(BytesTTS, "load_path", no)
    assert main(["speakers", "tiny"]) == 0
    assert capsys.readouterr().out.splitlines() == ["a\t\t"]


def test_speakers_describes_vctk_speakers(capsys: pytest.CaptureFixture[str]) -> None:  # the real catalogue and details
    assert main(["speakers", "en-vctk"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "p228\tfeatured\tFemale, 22 · English (England), Southern England"
    assert next(line for line in lines if line.startswith("ED\t")).endswith("\tnot in the VCTK speaker table")
    assert len(lines) == 109 and all(line.count("\t") == 2 for line in lines)


def test_speakers_on_an_ascii_stdout(monkeypatch: pytest.MonkeyPatch) -> None:  # Review Focus 6
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out, encoding="ascii"))
    assert main(["speakers", "en-vctk"]) == 0
    sys.stdout.flush()
    first = out.getvalue().decode("ascii").splitlines()[0]
    assert first == "p228\tfeatured\tFemale, 22 ? English (England), Southern England"


def test_download_one_and_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"a": _tiny(id="a"), "b": _tiny(id="b")})
    fetched: list[str] = []
    monkeypatch.setattr("bytestts.cli.ensure_voice", lambda voice_id, **k: fetched.append(voice_id) or tmp_path)
    assert main(["download", "b"]) == 0 and fetched == ["b"]
    assert main(["download", "--all"]) == 0 and fetched == ["b", "a", "b"]


def test_download_needs_one_target() -> None:
    assert main(["download"]) == 2 and main(["download", "a", "--all"]) == 2


def test_download_unknown_exit_3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {})
    assert main(["download", "nope"]) == 3


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0 and bytestts.__version__ in capsys.readouterr().out
    assert main(["--help"]) == 0 and main([]) == 2
