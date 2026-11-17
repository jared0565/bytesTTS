# SPDX-License-Identifier: MPL-2.0
"""Voice packaging (needs torch: it loads the generator and exports ONNX)."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from helpers import TEST_THREADS, TINY, edit_json

from bytestts import BytesTTS
from bytestts.catalog.manifest import VoiceManifest, verify_package

ROOT = Path(__file__).resolve().parents[2]


def load_tool(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pv = load_tool("tools/convert/package_voice.py")
fixture_tool = load_tool("tools/dev/make_tiny_fixture.py")
CREDITS = {"licence": "MIT", "data_licence": "MIT", "attribution": "x"}


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d: Path = tmp_path_factory.mktemp("converted") / "tiny"
    fixture_tool.make_converted(d)
    return d


@pytest.fixture(scope="module")
def converted_multi(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d: Path = tmp_path_factory.mktemp("converted") / "multi"
    fixture_tool.make_converted(d, speakers={"alice": 0, "bob": 2, "carol": 1})
    return d


def package(converted: Path, out: Path, **kw: Any) -> VoiceManifest:
    m: VoiceManifest = pv.package(
        converted, out, **{"voice_id": "t", "version": "1", "language": "en", **CREDITS, **kw}
    )
    return m


def test_package_from_tiny(converted: Path, tmp_path: Path) -> None:
    m = package(converted, tmp_path / "pkg")
    assert verify_package(tmp_path / "pkg") == m
    assert (m.id, m.origin, m.engine, m.sample_rate) == ("t", "catalog", "vits-onnx", 22050)


def test_package_reproduces_the_committed_fixture(converted: Path, tmp_path: Path) -> None:
    # the converted layout holds the fixture's own weights and config: packaging must give back the same bytes
    package(converted, tmp_path / "pkg")
    for name in ("model.onnx", "config.json"):
        assert (tmp_path / "pkg" / name).read_bytes() == (TINY / name).read_bytes(), name


def test_packaged_voice_speaks(converted: Path, tmp_path: Path) -> None:
    package(converted, tmp_path / "pkg")
    assert BytesTTS.load_path(tmp_path / "pkg", device="cpu", threads=TEST_THREADS).synthesize("Hi.").duration > 0


def test_speakers_are_carried(converted_multi: Path, tmp_path: Path) -> None:
    m = package(converted_multi, tmp_path / "pkg")
    assert m.speakers == ["alice", "carol", "bob"] and "speakers.json" in m.files  # listed in id order
    tts = BytesTTS.load_path(tmp_path / "pkg", device="cpu", threads=TEST_THREADS)
    assert tts.synthesize("Hi.", speaker="bob").duration > 0


def test_vocabulary_must_fit_the_embedding(converted: Path, tmp_path: Path) -> None:
    d = shutil.copytree(converted, tmp_path / "c")
    edit_json(d / "voice_config.json", lambda v: v["characters"].__setitem__("characters", "abc"))
    with pytest.raises(ValueError, match="vocabulary"):
        package(d, tmp_path / "pkg")


def test_language_must_agree(converted: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="language"):
        package(converted, tmp_path / "pkg", language="de")


def test_never_writes_into_a_non_empty_directory(converted: Path, tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "old.onnx").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        package(converted, tmp_path / "pkg")


def test_cli(converted: Path, tmp_path: Path) -> None:
    args = [str(converted), str(tmp_path / "pkg"), "--id", "t", "--version", "1", "--language", "en"]
    args += ["--licence", "MIT", "--data-licence", "MIT", "--attribution", "someone"]
    assert pv.main(args) == 0
    assert json.loads((tmp_path / "pkg" / "voice.json").read_bytes())["attribution"] == "someone"


def test_tools_never_torch_load() -> None:
    assert "torch.load" not in (ROOT / "tools" / "convert" / "package_voice.py").read_text("utf-8")
