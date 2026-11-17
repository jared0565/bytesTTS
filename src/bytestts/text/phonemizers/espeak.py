# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/utils/text/phonemizers/espeak_wrapper.py (and base.py)
"""espeak-ng as a subprocess. Text goes on stdin only, never in argv.

TTS-dev appended the text to argv (espeak_wrapper.py:188). Text starting with "-" was then parsed as
an option, and "-f<file>" made espeak read local files. On Windows, argv also reached espeak as
cp1252, which garbled non-ASCII text. P0.3 measured both. Here the text is UTF-8 bytes on stdin,
newline-terminated, because stdin drops the last character without the newline (P0.3).
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from bytestts.errors import FrontendUnavailable
from bytestts.text.punctuation import Punctuation

PINNED_VERSION = "1.52.0"
_TIMEOUT_S = 30
_VOICE_NAME = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8}){0,3}")
_LANG_FLAG = re.compile(r"\(.+?\)")
_VERSION = re.compile(r"text-to-speech:\s*(\S+)")

INSTALL_HINT = (
    f"espeak-ng {PINNED_VERSION} is required for this voice. "
    "Windows, no admin rights: download espeak-ng.msi from "
    f"https://github.com/espeak-ng/espeak-ng/releases/tag/{PINNED_VERSION}, run "
    r"`msiexec /a espeak-ng.msi /qn TARGETDIR=%LOCALAPPDATA%\bytestts-tools\espeak-ng`, then set "
    r"BYTESTTS_ESPEAK to ...\eSpeak NG\espeak-ng.exe and BYTESTTS_ESPEAK_DATA to ...\eSpeak NG. "
    "Linux: install the espeak-ng package, or build the tag from source."
)


@dataclass(frozen=True)
class EspeakInstall:
    executable: Path
    data_path: Path | None  # the directory that CONTAINS espeak-ng-data; None = espeak-ng's built-in default


def _data_path_for(executable: Path) -> Path | None:
    env = os.environ.get("BYTESTTS_ESPEAK_DATA")
    if env:
        return Path(env)
    beside = executable.parent
    return beside if (beside / "espeak-ng-data").is_dir() else None  # the msiexec /a layout


def find_espeak(executable: str | os.PathLike[str] | None = None) -> EspeakInstall:
    """The argument, then $BYTESTTS_ESPEAK (+ $BYTESTTS_ESPEAK_DATA), then espeak-ng on PATH."""
    for candidate in (executable, os.environ.get("BYTESTTS_ESPEAK") or None):
        if candidate is not None:
            exe = Path(candidate)
            if not exe.is_file():
                raise FrontendUnavailable(f"espeak-ng executable not found at {exe}. {INSTALL_HINT}")
            return EspeakInstall(exe, _data_path_for(exe))
    found = _on_path()
    if found is None:
        raise FrontendUnavailable(f"espeak-ng not found. {INSTALL_HINT}")
    return EspeakInstall(found, _data_path_for(found))


def _on_path() -> Path | None:
    """espeak-ng from the absolute PATH entries only. Windows (and shutil.which there) searches the current
    directory first, and POSIX reads an empty or relative entry as it: either way a planted espeak-ng in a
    downloaded folder would run. On Windows only espeak-ng.exe counts; a .bat or .cmd runs through cmd.exe."""
    name = "espeak-ng.exe" if os.name == "nt" else "espeak-ng"
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or not os.path.isabs(entry):
            continue
        candidate = Path(entry) / name
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return candidate
    return None


class EspeakPhonemizer:
    """IPA phonemes from espeak-ng, with punctuation kept as TTS-dev's BasePhonemizer keeps it."""

    def __init__(self, language: str, *, install: EspeakInstall | None = None, keep_puncs: bool = True) -> None:
        if language == "en":
            language = "en-us"  # as TTS-dev does
        if not _VOICE_NAME.fullmatch(language):
            raise ValueError(f"not an espeak-ng voice name: {language!r}")  # it goes into argv as -v <language>
        self.language = language
        self.install = install if install is not None else find_espeak()
        if self.install.data_path is not None and not (self.install.data_path / "espeak-ng-data").is_dir():
            # espeak-ng 1.52.0 crashes (0xC0000005) on a bogus data path instead of reporting it (P0.3)
            raise FrontendUnavailable(f"no espeak-ng-data under {self.install.data_path}. {INSTALL_HINT}")
        self.keep_puncs = keep_puncs
        self._punctuator = Punctuation()

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.install.data_path is not None:
            env["ESPEAK_DATA_PATH"] = str(self.install.data_path)
        return env

    def _call(self, args: list[str], stdin: bytes | None) -> str:
        cmd = [str(self.install.executable), *args]
        try:
            done = subprocess.run(  # noqa: S603 -- fixed argv; user text only ever travels on stdin
                cmd, input=stdin, capture_output=True, check=True, timeout=_TIMEOUT_S, env=self._env()
            )
        except subprocess.CalledProcessError as e:
            raise FrontendUnavailable(f"espeak-ng failed (exit code {e.returncode}). {INSTALL_HINT}") from e
        except (OSError, subprocess.TimeoutExpired) as e:
            raise FrontendUnavailable(f"espeak-ng could not run: {e}. {INSTALL_HINT}") from e
        return done.stdout.decode("utf-8")

    @functools.cached_property
    def version(self) -> str:
        """The dotted version from `--version`, e.g. "1.52.0"."""
        m = _VERSION.search(self._call(["--version"], None))
        if m is None:
            raise FrontendUnavailable("could not read the espeak-ng version")
        return m.group(1)

    def _run_espeak(self, text: str) -> str:
        """Raw stdout for one stretch of text."""
        return self._call(["-q", "-b", "1", "--ipa=1", "-v", self.language, "--stdin"], text.encode("utf-8") + b"\n")

    def _phonemize_part(self, text: str, separator: str) -> str:
        phonemes = ""
        for line in self._run_espeak(text).splitlines():  # espeak_wrapper.py:189-206
            phonemes += _LANG_FLAG.sub("", line.strip()).strip()
        return phonemes.replace("_", separator)

    def phonemize(self, text: str, separator: str = "") -> str:
        text = text.strip()
        if self.keep_puncs:
            parts, puncs = self._punctuator.strip_to_restore(text)
            return self._punctuator.restore([self._phonemize_part(t, separator) for t in parts], puncs)[0]
        return self._phonemize_part(self._punctuator.strip(text), separator)
