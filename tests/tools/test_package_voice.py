# SPDX-License-Identifier: MPL-2.0
"""Voice packaging (needs torch: it loads the generator and exports ONNX)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
from bytestts.errors import IntegrityError, VoiceNotFound

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


def _zero_noise_wav(pkg: Path, speaker: str) -> np.ndarray:
    from bytestts.engines.base import SynthesisOptions, VoiceRef
    from bytestts.engines.vits_onnx import VitsDefaults, VitsOnnxEngine

    config = json.loads((pkg / "config.json").read_bytes())
    speakers = json.loads((pkg / "speakers.json").read_bytes())
    defaults = VitsDefaults.from_config(config)
    engine = VitsOnnxEngine(pkg / "model.onnx", defaults, device="cpu", threads=TEST_THREADS, speakers=speakers)
    zero = SynthesisOptions(noise_scale=0.0, noise_scale_dp=0.0)
    wav, _ = engine.synthesize_with_durations([3, 5, 7, 2, 9], VoiceRef("t", speaker=speaker), zero)
    return np.asarray(wav)


def test_a_renamed_speaker_renders_its_original_row(converted_multi: Path, tmp_path: Path) -> None:
    package(converted_multi, tmp_path / "a")
    m = package(converted_multi, tmp_path / "b", rename_speakers={"bob": "robert"})
    assert m.speakers == ["alice", "carol", "robert"]
    assert (tmp_path / "b" / "model.onnx").read_bytes() == (tmp_path / "a" / "model.onnx").read_bytes()
    robert = _zero_noise_wav(tmp_path / "b", "robert")
    assert np.array_equal(robert, _zero_noise_wav(tmp_path / "a", "bob"))
    assert not np.array_equal(robert, _zero_noise_wav(tmp_path / "a", "alice"))  # control: the rows do differ
    with pytest.raises(VoiceNotFound):  # the old name is gone
        BytesTTS.load_path(tmp_path / "b", device="cpu", threads=TEST_THREADS).synthesize("Hi.", speaker="bob")


def _cli_args(converted: Path, out: Path, pairs: list[str]) -> list[str]:
    args = [str(converted), str(out), "--id", "t", "--version", "1", "--language", "en"]
    args += ["--licence", "MIT", "--data-licence", "MIT", "--attribution", "x"]
    for pair in pairs:
        args += ["--rename-speaker", pair]
    return args


def test_cli_renames_a_speaker(converted_multi: Path, tmp_path: Path) -> None:
    assert pv.main(_cli_args(converted_multi, tmp_path / "pkg", ["bob=robert"])) == 0
    assert json.loads((tmp_path / "pkg" / "speakers.json").read_bytes()) == {"alice": 0, "carol": 1, "robert": 2}


@pytest.mark.parametrize(
    ("voice", "pairs", "reason"),
    [
        ("converted_multi", ["zed=z"], "no speaker named 'zed'"),
        ("converted_multi", ["bob=alice"], "already a speaker's name"),
        ("converted_multi", ["bob"], "not OLD=NEW"),
        ("converted_multi", ["bob="], "not a usable speaker name"),
        ("converted_multi", ["bob= robert"], "not a usable speaker name"),
        ("converted_multi", ["bob=x", "bob=y"], "twice"),
        ("converted", ["bob=robert"], "single speaker"),
    ],
)
def test_rename_refuses_a_bad_pair(
    voice: str,
    pairs: list[str],
    reason: str,
    request: pytest.FixtureRequest,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:  # exit 2 alone would also pass with no such option at all: argparse refuses unknown flags with 2
    with pytest.raises(SystemExit) as refused:
        pv.main(_cli_args(request.getfixturevalue(voice), tmp_path / "pkg", pairs))
    assert refused.value.code == 2 and not (tmp_path / "pkg").exists()
    assert reason in capsys.readouterr().err


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


def test_cli_runs_torch_single_threaded() -> None:  # intra- AND inter-op: all-core load resets the development PC
    # a fresh process: this one's torch is already pinned by the test conftest, so checking it here proves nothing
    probe = (
        "import importlib.util, sys, torch\n"
        f"spec = importlib.util.spec_from_file_location('pv', {str(ROOT / 'tools/convert/package_voice.py')!r})\n"
        "pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)\n"
        "try:\n    pv.main(['--help'])\nexcept SystemExit:\n    pass\n"
        "print(torch.get_num_threads(), torch.get_num_interop_threads())\n"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}  # not the conftest's caps: main must pin
    out = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter and a script built here
        [sys.executable, "-P", "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.split()[-2:] == ["1", "1"], out.stdout


def test_tools_never_torch_load() -> None:
    assert "torch.load" not in (ROOT / "tools" / "convert" / "package_voice.py").read_text("utf-8")


def test_one_speaker_language_voice_packages_and_speaks(tmp_path: Path) -> None:
    # The CSS10 layout end to end: the real converter, the packager, then synthesis with no speaker named.
    cc = load_tool("tools/convert/convert_coqui.py")
    args = {
        **fixture_tool.TINY_ARGS,
        "use_language_embedding": True,
        "num_languages": 1,
        "embedded_language_dim": 4,
        "use_speaker_embedding": True,
        "num_speakers": 1,
        "speaker_embedding_channels": 8,
    }
    model = fixture_tool._tiny_model(args)
    cfg = fixture_tool.coqui_config({**args, "num_speakers": 0})
    cc.convert_state(
        model.state_dict(), cfg, language="es", speaker_ids={"tux": 0}, language_ids={"es": 0}, out=tmp_path / "c"
    )
    m = package(tmp_path / "c", tmp_path / "pkg", language="es")
    assert m.speakers == ["tux"] and "speakers.json" in m.files
    tts = BytesTTS.load_path(tmp_path / "pkg", device="cpu", threads=TEST_THREADS)
    assert tts.info.speakers == ["tux"]
    assert tts.synthesize("Hola.").duration > 0  # no speaker named: the only one
