# SPDX-License-Identifier: MPL-2.0
"""convert_coqui's archive handling and host refusal (no torch, no pickle: those run only in the sandbox)."""

import hashlib
import importlib.util
import json
import re
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
# The converter reads only os.name. Swap its own `os` binding, never the real module's attribute: on Python 3.11
# a patched os.name makes every new Path a WindowsPath, which POSIX refuses, pytest's failure report included.
ON_WINDOWS = SimpleNamespace(name="nt")


def load_tool(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cc = load_tool("tools/convert/convert_coqui.py")


def make_zip(path: Path, names: list[str]) -> zipfile.ZipFile:
    with zipfile.ZipFile(path, "w") as z:
        for name in names:
            z.writestr(name, b"x")
    return zipfile.ZipFile(path)


def test_members_skip_macos_resource_forks(tmp_path: Path) -> None:  # the css10 archives' shape (P0.5)
    names = ["v/._v", "v/config.json", "v/._config.json", "v/speaker_ids.json", "v/._speaker_ids.json"]
    names += ["v/model_file.pth.tar", "v/._model_file.pth.tar", "__MACOSX/v/config.json", "v/language_ids.json"]
    with make_zip(tmp_path / "a.zip", names) as z:
        assert cc._members(z) == {
            "checkpoint": "v/model_file.pth.tar",
            "config.json": "v/config.json",
            "speaker_ids.json": "v/speaker_ids.json",
        }


def test_members_refuse_two_checkpoints(tmp_path: Path) -> None:
    with make_zip(tmp_path / "a.zip", ["v/config.json", "v/a.pth", "v/b.pth"]) as z, pytest.raises(cc.Refused) as e:
        cc._members(z)
    assert e.value.code == 4


@pytest.mark.parametrize("names", [["v/model_file.pth"], ["v/config.json"]])
def test_members_need_a_checkpoint_and_a_config(tmp_path: Path, names: list[str]) -> None:
    with make_zip(tmp_path / "a.zip", names) as z, pytest.raises(cc.Refused) as e:
        cc._members(z)
    assert e.value.code == 4


def sources_for(archive: Path, *, sha256: str | None = None, size: int | None = None) -> Path:
    data = archive.read_bytes()
    entry = {
        "url": f"https://example.invalid/{archive.name}",
        "sha256": sha256 or hashlib.sha256(data).hexdigest(),
        "size": len(data) if size is None else size,
    }
    path = archive.with_suffix(".sources.json")
    path.write_text(json.dumps({"en/x/vits": entry}), "utf-8")
    return path


def test_source_pins_the_archive(tmp_path: Path) -> None:
    make_zip(tmp_path / "x.zip", ["v/config.json"]).close()
    assert cc._source(tmp_path / "x.zip", sources_for(tmp_path / "x.zip"))[0] == "en/x/vits"


@pytest.mark.parametrize("pin", [{"sha256": "0" * 64}, {"size": 1}])
def test_source_refuses_a_changed_archive(tmp_path: Path, pin: dict[str, object]) -> None:
    make_zip(tmp_path / "x.zip", ["v/config.json"]).close()
    sources = sources_for(tmp_path / "x.zip", **pin)  # type: ignore[arg-type]
    with pytest.raises(cc.Refused, match="differs") as e:
        cc._source(tmp_path / "x.zip", sources)
    assert e.value.code == 4


def test_source_refuses_an_unlisted_archive(tmp_path: Path) -> None:
    make_zip(tmp_path / "x.zip", ["v/config.json"]).close()
    sources = sources_for(tmp_path / "x.zip")
    make_zip(tmp_path / "y.zip", ["v/config.json"]).close()
    with pytest.raises(cc.Refused, match="not listed"):
        cc._source(tmp_path / "y.zip", sources)


def test_refuses_to_run_on_windows(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def never(*a: object, **k: object) -> None:
        raise AssertionError("convert ran")

    monkeypatch.setattr(cc, "convert", never)
    monkeypatch.setattr(cc, "os", ON_WINDOWS)
    assert cc.main(["a.zip", "out"]) == 2
    assert "sandbox" in capsys.readouterr().err


def test_shipped_sources_are_pinned() -> None:
    sources = json.loads((ROOT / "tools" / "convert" / "sources.json").read_bytes())
    assert len(sources) == 8
    for entry in sources.values():
        assert entry["url"].startswith("https://github.com/coqui-ai/TTS/releases/download/")
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) and entry["size"] > 0


@pytest.mark.parametrize("allow_full_load", [False, True])
def test_the_unpickler_itself_refuses_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, allow_full_load: bool
) -> None:  # not only main(): any caller of _load, a fuzzer's included, is refused before torch is imported
    monkeypatch.setattr(cc, "os", ON_WINDOWS)
    (tmp_path / "x.pth").write_bytes(b"not a pickle")
    with pytest.raises(cc.Refused, match="sandbox") as e:
        cc._load(tmp_path / "x.pth", allow_full_load)
    assert e.value.code == 2
