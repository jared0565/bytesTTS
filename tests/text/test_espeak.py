# SPDX-License-Identifier: MPL-2.0
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from golden import VOICE_CFG, trace
from helpers import require_espeak

from bytestts.errors import FrontendUnavailable
from bytestts.text.phonemizers import espeak as espeak_mod
from bytestts.text.phonemizers.espeak import EspeakInstall, EspeakPhonemizer, find_espeak

ESPEAK_EXPECTED: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "espeak" / "expected.json").read_text("utf-8")
)
FAKE = EspeakInstall(Path("espeak-ng"), None)


@pytest.mark.espeak
def test_matches_reference_phonemes() -> None:
    require_espeak(trace["espeak_version"])  # golden phonemes are only valid for the pinned tag
    p = EspeakPhonemizer(VOICE_CFG["phoneme_language"])
    for s in trace["sentences"]:
        assert p.phonemize(s["cleaned"], separator="") == s["phonemes"], s["cleaned"]


def test_text_never_in_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], bytes]] = []  # one espeak call per punctuation-separated part

    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((cmd, kw["input"]))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"h@l'oU\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    EspeakPhonemizer("en-us", install=FAKE).phonemize("-w/tmp/x.wav hello")
    assert calls
    for cmd, stdin in calls:
        assert all("hello" not in a and "-w/tmp" not in a and "wav" not in a for a in cmd)
        assert cmd[-1] == "--stdin" and "-q" in cmd
        assert stdin.endswith(b"\n")  # UTF-8 on stdin, newline-terminated (P0.3)
    assert [stdin for _, stdin in calls] == [b"-w/tmp/x\n", b"wav hello\n"]  # "." is punctuation


@pytest.mark.espeak
def test_option_text_is_spoken_literally(tmp_path: Path) -> None:  # replaces the plan's vacuous -w probe (P0.3)
    install = require_espeak(ESPEAK_EXPECTED["espeak_version"])
    secret = tmp_path / "secret.txt"
    secret.write_text("zebra crossing\n", "utf-8")
    env = None
    if install.data_path is not None:
        env = {**espeak_mod.os.environ, "ESPEAK_DATA_PATH": str(install.data_path)}
    control = subprocess.run(  # noqa: S603 -- the control runs espeak via argv on purpose, in the test only
        [str(install.executable), "-q", "-b", "1", "--ipa=1", "-v", "en-us", "-f", str(secret)],
        capture_output=True,
        timeout=30,
        env=env,
    ).stdout.decode("utf-8")
    assert "z_ˈiː_b_ɹ_ə" in control  # the probe works: -f really reads the file
    out = EspeakPhonemizer("en-us", install=install).phonemize("-f" + str(secret), separator="_")
    assert out and "z_ˈiː_b_ɹ_ə" not in out and "ɹ_ˈɔ_s_ɪ_ŋ" not in out


@pytest.mark.espeak
@pytest.mark.parametrize("case", ESPEAK_EXPECTED["cases"], ids=lambda c: f"{c['voice']}:{c['text'][:20]}")
def test_non_ascii_matches_recorded_stdin_output(case: dict[str, str]) -> None:
    install = require_espeak(ESPEAK_EXPECTED["espeak_version"])
    raw = EspeakPhonemizer(case["voice"], install=install)._run_espeak(case["text"])
    strip = lambda s: s.replace("\r\n", "\n").strip("\n")  # noqa: E731
    assert strip(raw) == strip(case["raw"])  # exact: mojibake cannot pass


def test_require_espeak_fails_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BYTESTTS_REQUIRE_ESPEAK", "1")
    with pytest.raises(pytest.fail.Exception):
        require_espeak("0.0.0-never")


def test_missing_espeak_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("BYTESTTS_ESPEAK", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)  # this PC has espeak-ng in the documented folder
    with pytest.raises(FrontendUnavailable, match="espeak-ng"):
        find_espeak()


def _stub(directory: Path, name: str = "espeak-ng.exe" if os.name == "nt" else "espeak-ng") -> Path:
    """An empty file named like espeak-ng. Never run: these tests only check which file is found."""
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / name
    exe.write_bytes(b"")
    exe.chmod(0o755)
    return exe


def _path_only(monkeypatch: pytest.MonkeyPatch, *entries: str) -> None:
    monkeypatch.delenv("BYTESTTS_ESPEAK", raising=False)
    monkeypatch.delenv("BYTESTTS_ESPEAK_DATA", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)  # no documented-folder fallback unless a test sets it
    monkeypatch.setenv("PATH", os.pathsep.join(entries))


def test_lookup_never_takes_espeak_from_the_current_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Windows searches the current directory first (shutil.which too), so a planted espeak-ng.exe in a downloaded
    # folder would run with the user's rights. POSIX reads an empty PATH entry, or ".", as the current directory.
    _stub(tmp_path / "cwd")
    (tmp_path / "empty").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    _path_only(monkeypatch, "", ".", str(tmp_path / "empty"))
    with pytest.raises(FrontendUnavailable, match="not found"):
        find_espeak()


def test_lookup_finds_espeak_in_an_absolute_path_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = _stub(tmp_path / "bin")
    monkeypatch.chdir(tmp_path)
    _path_only(monkeypatch, ".", str(tmp_path / "bin"))
    assert find_espeak().executable == exe


@pytest.mark.skipif(os.name != "nt", reason="Windows file types")
def test_lookup_ignores_batch_files_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub(tmp_path / "bin", "espeak-ng.bat")  # a .bat or .cmd runs through cmd.exe's own parsing
    _stub(tmp_path / "bin", "espeak-ng.cmd")
    _path_only(monkeypatch, str(tmp_path / "bin"))
    with pytest.raises(FrontendUnavailable, match="not found"):
        find_espeak()


@pytest.mark.skipif(os.name != "nt", reason="the documented folder is Windows-only")
def test_documented_folder_is_the_last_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = _stub(tmp_path / "bytestts-tools" / "espeak-ng" / "eSpeak NG")
    (tmp_path / "empty").mkdir()
    _path_only(monkeypatch, str(tmp_path / "empty"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert find_espeak().executable == exe


@pytest.mark.skipif(os.name != "nt", reason="the documented folder is Windows-only")
def test_path_and_env_still_beat_the_documented_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub(tmp_path / "bytestts-tools" / "espeak-ng" / "eSpeak NG")
    on_path = _stub(tmp_path / "bin")
    _path_only(monkeypatch, str(tmp_path / "bin"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert find_espeak().executable == on_path
    env_exe = _stub(tmp_path / "envdir")
    monkeypatch.setenv("BYTESTTS_ESPEAK", str(env_exe))
    assert find_espeak().executable == env_exe


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_lookup_ignores_a_file_that_is_not_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub(tmp_path / "bin").chmod(0o644)
    _path_only(monkeypatch, str(tmp_path / "bin"))
    with pytest.raises(FrontendUnavailable, match="not found"):
        find_espeak()


def test_data_path_without_espeak_ng_data_raises(tmp_path: Path) -> None:  # a bogus data path crashes espeak (P0.3)
    exe = tmp_path / "espeak-ng.exe"
    exe.write_bytes(b"")
    with pytest.raises(FrontendUnavailable):
        EspeakPhonemizer("en-us", install=EspeakInstall(exe, tmp_path / "nowhere"))


def test_crash_exit_maps_to_frontend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def crashed(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(-1073741819, cmd)  # 0xC0000005, what espeak does on bad data

    monkeypatch.setattr(subprocess, "run", crashed)
    with pytest.raises(FrontendUnavailable):
        EspeakPhonemizer("en-us", install=FAKE).phonemize("hello")


def test_en_maps_to_en_us() -> None:
    assert EspeakPhonemizer("en", install=FAKE).language == "en-us"


@pytest.mark.parametrize("language", ["-f/etc/passwd", "en us", "en;rm", "", "x" * 40])
def test_language_must_be_a_voice_name(language: str) -> None:  # it goes into argv as `-v <language>`
    with pytest.raises(ValueError):
        EspeakPhonemizer(language, install=FAKE)
