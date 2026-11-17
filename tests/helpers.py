# SPDX-License-Identifier: MPL-2.0
"""Shared test helpers. Imported as `helpers` (pyproject pythonpath = tests), never from conftest.py:
under --import-mode=importlib, importing from conftest.py creates a second copy of the module."""

import os

import pytest

from bytestts.errors import FrontendUnavailable
from bytestts.text.phonemizers.espeak import EspeakInstall, EspeakPhonemizer, find_espeak


def require_espeak(expected_version: str | None) -> EspeakInstall:
    """The espeak-ng install the tests should use. Skips when it is missing or not `expected_version`,
    but FAILS instead when BYTESTTS_REQUIRE_ESPEAK=1 (CI, the matrix), so a gap can't pass unseen."""
    required = os.environ.get("BYTESTTS_REQUIRE_ESPEAK") == "1"
    try:
        install = find_espeak()
        version = EspeakPhonemizer("en-us", install=install).version
    except FrontendUnavailable as e:
        reason = f"espeak-ng unavailable: {e}"
        if required:
            pytest.fail(reason)
        pytest.skip(reason)
    if expected_version is not None and version != expected_version:
        reason = f"espeak-ng {version} found, tests need the pinned {expected_version}"
        if required:
            pytest.fail(reason)
        pytest.skip(reason)
    return install
