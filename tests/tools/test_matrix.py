# SPDX-License-Identifier: MPL-2.0
"""tools/dev/matrix.py reports the espeak-ng the tests will use, found the way the library finds it."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from helpers import require_espeak

from bytestts.text.phonemizers.espeak import EspeakPhonemizer

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("matrix", ROOT / "tools" / "dev" / "matrix.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load()


@pytest.mark.espeak
def test_matrix_reports_the_espeak_the_tests_use() -> None:
    install = require_espeak(None)  # with the msiexec layout, the data path comes from beside the executable
    assert matrix.espeak_version() == EspeakPhonemizer("en-us", install=install).version


def test_matrix_reports_a_missing_espeak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BYTESTTS_ESPEAK", str(tmp_path / "no-espeak-ng.exe"))
    assert matrix.espeak_version().startswith("not found")
