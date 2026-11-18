# SPDX-License-Identifier: MPL-2.0
import importlib.util
from email.message import Message
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_licences", ROOT / "tools" / "dev" / "check_licences.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cl = _load_checker()
classify = cl.classify
licence_of = cl.licence_of


class FakeDist:
    """Duck-types the part of importlib.metadata.Distribution the checker reads: .metadata."""

    def __init__(
        self,
        fields: dict[str, Any] | None = None,
        *,
        name: str = "fake-dist",
        lic: str | None = None,
        version: str = "1.0",
    ) -> None:
        msg = Message()
        msg["Name"] = name
        msg["Version"] = version
        for key, value in (fields or {}).items():
            for v in value if isinstance(value, list) else [value]:
                msg[key] = v
        if lic is not None:
            msg["License"] = lic
        self.metadata = msg


def test_classify() -> None:
    assert classify("MIT") == "allowed"
    assert classify("Apache-2.0") == "allowed"
    assert classify("LGPL-2.1-or-later") == "lgpl"
    assert classify("GPL-3.0-only") == "denied"
    assert classify("") == "unknown"


def test_classify_spdx_expressions() -> None:
    assert classify("Apache-2.0 OR BSD-2-Clause") == "allowed"  # packaging's own expression
    assert classify("MIT OR GPL-3.0-only") == "allowed"
    assert classify("MIT AND GPL-3.0-only") == "denied"
    assert classify("(MIT OR Apache-2.0) AND BSD-3-Clause") == "allowed"
    assert classify("Apache-2.0 WITH LLVM-exception") == "allowed"


@pytest.mark.parametrize(
    "expr", ["MIT AND", "(MIT", "MIT OR OR BSD-3-Clause", "LicenseRef-Proprietary", "Other/Proprietary"]
)
def test_classify_malformed_or_unlisted_is_unknown(expr: str) -> None:
    assert classify(expr) == "unknown"


def test_licence_of_falls_back_to_classifier() -> None:
    d = FakeDist({"Classifier": ["License :: OSI Approved :: BSD License"], "License": "<full licence text>\n..."})
    assert classify(licence_of(d)) == "allowed"


def test_licence_of_prefers_license_expression() -> None:
    d = FakeDist({"License-Expression": "GPL-3.0-only", "Classifier": ["License :: OSI Approved :: MIT License"]})
    assert classify(licence_of(d)) == "denied"


def test_licence_of_ignores_long_licence_text() -> None:
    assert (
        licence_of(FakeDist({"License": "MIT License\n\nCopyright (c) someone. Permission is hereby granted..."})) == ""
    )


def test_nvidia_exception_only_for_listed_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cl, "NVIDIA_EXCEPTIONS", frozenset({"nvidia-cudnn-cu12"}))
    assert cl.violations([FakeDist(name="nvidia-cudnn-cu12", lic="Other/Proprietary")]) == []
    assert cl.violations([FakeDist(name="nvidia-new-thing", lic="Other/Proprietary")]) != []


def test_nvidia_exceptions_start_empty() -> None:
    assert frozenset() == cl.NVIDIA_EXCEPTIONS


def test_reviewed_exception_holds_only_for_the_reviewed_expression(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cl, "REVIEWED", {"some-dist": "MIT AND Zlib"})
    assert cl.violations([FakeDist({"License-Expression": "MIT AND Zlib"}, name="Some_Dist")]) == []
    assert cl.violations([FakeDist({"License-Expression": "MIT AND Zlib AND GPL-3.0-only"}, name="some-dist")]) != []
    assert cl.violations([FakeDist({"License-Expression": "MIT AND Zlib"}, name="other-dist")]) != []


def test_reviewed_metadata_free_dist_passes_at_its_version() -> None:
    assert cl.violations([FakeDist(name="clr-loader", version="0.3.1")]) == []


def test_reviewed_metadata_free_dist_fails_at_another_version() -> None:
    assert len(cl.violations([FakeDist(name="clr-loader", version="0.3.2")])) == 1


def test_metadata_free_exception_needs_empty_metadata() -> None:  # a licence that later appears is judged normally
    assert len(cl.violations([FakeDist(name="clr-loader", version="0.3.1", lic="GPL-3.0-only")])) == 1
