# SPDX-License-Identifier: MPL-2.0
"""Voice packaging (needs torch: it loads the generator and exports ONNX)."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnx
import pytest
from helpers import TEST_THREADS, TINY, edit_json
from onnx import numpy_helper

from bytestts import BytesTTS
from bytestts.catalog.manifest import VoiceManifest, verify_package
from bytestts.errors import IntegrityError

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


@pytest.fixture(scope="module")
def fresh_tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny fixture as the fixture tool writes it here and now, from the same seeds."""
    d: Path = tmp_path_factory.mktemp("fresh") / "tiny"
    fixture_tool.make_tiny(d, with_generator=False)
    return d


def package(converted: Path, out: Path, **kw: Any) -> VoiceManifest:
    m: VoiceManifest = pv.package(
        converted, out, **{"voice_id": "t", "version": "1", "language": "en", **CREDITS, **kw}
    )
    return m


def _without_weights(model: onnx.ModelProto) -> bytes:
    bare = onnx.ModelProto.FromString(model.SerializeToString())
    for t in bare.graph.initializer:  # names, shapes and order stay; only the values go
        t.ClearField("raw_data")
        t.ClearField("float_data")
    return bare.SerializeToString()


def test_package_from_tiny(converted: Path, tmp_path: Path) -> None:
    m = package(converted, tmp_path / "pkg")
    assert verify_package(tmp_path / "pkg") == m
    assert (m.id, m.origin, m.engine, m.sample_rate) == ("t", "catalog", "vits-onnx", 22050)


def test_package_gives_the_fixture_tools_bytes(converted: Path, fresh_tiny: Path, tmp_path: Path) -> None:
    # the converted layout holds the fixture's own weights and config: packaging must export the same bytes
    package(converted, tmp_path / "pkg")
    for name in ("model.onnx", "config.json"):
        assert (tmp_path / "pkg" / name).read_bytes() == (fresh_tiny / name).read_bytes(), name


def test_committed_fixture_is_what_the_tools_write(fresh_tiny: Path) -> None:
    # The committed bytes are one platform's: torch's seeded weights and its weight-norm folding differ in the last
    # bits between its Windows and Linux builds (measured at most 6e-8, Task 17 Step 11). All else must be exact.
    assert (fresh_tiny / "config.json").read_bytes() == (TINY / "config.json").read_bytes()
    fresh, committed = onnx.load(str(fresh_tiny / "model.onnx")), onnx.load(str(TINY / "model.onnx"))
    assert _without_weights(fresh) == _without_weights(committed)
    for a, b in zip(fresh.graph.initializer, committed.graph.initializer, strict=True):
        x, y = numpy_helper.to_array(a), numpy_helper.to_array(b)
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-6, err_msg=a.name)


def test_packaged_voice_speaks(converted: Path, tmp_path: Path) -> None:
    package(converted, tmp_path / "pkg")
    assert BytesTTS.load_path(tmp_path / "pkg", device="cpu", threads=TEST_THREADS).synthesize("Hi.").duration > 0


def test_speakers_are_carried(converted_multi: Path, tmp_path: Path) -> None:
    m = package(converted_multi, tmp_path / "pkg")
    assert m.speakers == ["alice", "carol", "bob"] and "speakers.json" in m.files  # listed in id order
    tts = BytesTTS.load_path(tmp_path / "pkg", device="cpu", threads=TEST_THREADS)
    assert tts.synthesize("Hi.", speaker="bob").duration > 0


def test_a_multi_speaker_model_needs_its_speakers(converted_multi: Path, tmp_path: Path) -> None:
    # here because only this file builds a multi-speaker model: a package that drops its speakers is broken
    package(converted_multi, tmp_path / "pkg")
    edit_json(tmp_path / "pkg" / "voice.json", lambda m: m.update(speakers=[], featured=[]))
    with pytest.raises(IntegrityError, match="speaker id"):
        BytesTTS.load_path(tmp_path / "pkg", device="cpu", threads=TEST_THREADS)


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
