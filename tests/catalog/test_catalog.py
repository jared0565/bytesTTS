# SPDX-License-Identifier: MPL-2.0
import json
from pathlib import Path
from typing import Any

import pytest

from bytestts.catalog import catalog
from bytestts.catalog.catalog import VoiceInfo, list_voices, load_catalog, parse_catalog
from bytestts.catalog.download import ensure_voice
from bytestts.catalog.manifest import FileEntry, VoiceManifest, verify_package
from bytestts.errors import IntegrityError

SHA = "a" * 64
URL = "https://huggingface.co/owner/bytestts-voices/resolve/" + "0123456789abcdef" * 2 + "01234567/en/model.onnx"


def M(**kw: Any) -> VoiceManifest:
    base: dict[str, Any] = {
        "id": "v",
        "version": "1.0.0",
        "language": "en",
        "engine": "vits-onnx",
        "sample_rate": 22050,
        "licence": "MIT",
        "data_licence": "CC0-1.0",
        "attribution": "someone",
        "origin": "catalog",
        "files": {"model.onnx": FileEntry(SHA, 10, URL), "config.json": FileEntry(SHA, 5, URL)},
        "speakers": [],
        "featured": [],
    }
    return VoiceManifest(**{**base, **kw})


def entry(**kw: Any) -> dict[str, Any]:
    return M(**kw).to_dict()


def test_shipped_catalogue_loads() -> None:
    voices = load_catalog()  # the packaged voices.json, through importlib.resources
    assert all(isinstance(m, VoiceManifest) and m.id == key for key, m in voices.items())


def test_en_ljspeech_ships_in_the_catalogue() -> None:
    m = load_catalog()["en-ljspeech"]
    assert (m.language, m.engine, m.origin, m.sample_rate) == ("en", "vits-onnx", "catalog", 22050)
    pinned = "https://huggingface.co/fbmacagba/bytestts-en-ljspeech/resolve/8ec2c68f8187a2d68fbf97c657b2ccf4f73fbfb0/"
    urls = {name: entry.url for name, entry in m.files.items()}
    assert urls == {name: pinned + name for name in ("model.onnx", "config.json")}


@pytest.mark.network
@pytest.mark.parametrize("voice_id", sorted(load_catalog()))
def test_shipped_voice_downloads_and_verifies(voice_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real download of every catalogue voice (maintainers: `pytest -m network`; en-ljspeech is 114 MB)."""
    monkeypatch.setenv("BYTESTTS_HOME", str(tmp_path))
    directory = ensure_voice(voice_id)
    assert verify_package(directory) == load_catalog()[voice_id]


def test_list_voices_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"a": M(id="a", language="en"), "b": M(id="b", language="de")})
    assert [v.id for v in list_voices("de")] == ["b"]
    assert [v.id for v in list_voices()] == ["a", "b"]


def test_language_matches_regional_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    voices = {"a": M(id="a", language="en-gb"), "b": M(id="b", language="en"), "c": M(id="c", language="eng")}
    monkeypatch.setattr(catalog, "load_catalog", lambda: voices)
    assert [v.id for v in list_voices("en")] == ["a", "b"]  # not "eng": a prefix of the code, not of the tag
    assert [v.id for v in list_voices("EN-GB")] == ["a"]


def test_featured_voices_first_then_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    voices = {
        "c": M(id="c"),
        "b": M(id="b", speakers=["p1", "p2"], featured=["p2"]),
        "a": M(id="a"),
    }
    monkeypatch.setattr(catalog, "load_catalog", lambda: voices)
    assert [v.id for v in list_voices()] == ["b", "a", "c"]


def test_voice_info_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"v": M(speakers=["p1", "p2"], featured=["p2"])})
    (info,) = list_voices()
    assert info == VoiceInfo(
        id="v",
        language="en",
        speakers=["p1", "p2"],
        featured=["p2"],
        licence="MIT",
        data_licence="CC0-1.0",
        attribution="someone",
        sample_rate=22050,
        size_bytes=15,
    )


def _catalog(*entries: dict[str, Any]) -> str:
    return json.dumps({"voices": list(entries)})


def test_parse_catalog() -> None:
    voices = parse_catalog(_catalog(entry(id="a"), entry(id="b", language="de")))
    assert sorted(voices) == ["a", "b"] and voices["b"].language == "de"


@pytest.mark.parametrize(
    "text",
    [
        _catalog(entry(id="a"), entry(id="a")),  # duplicate id
        _catalog(entry(origin="fine-tuned")),  # the catalogue ships no fine-tuned voices
        _catalog(entry(files={"model.onnx": FileEntry(SHA, 10), "config.json": FileEntry(SHA, 5, URL)})),  # no url
        _catalog(entry(files={"model.onnx": FileEntry(SHA, 10, URL.replace("https", "http"))})),
        _catalog(
            entry(files={"model.onnx": FileEntry(SHA, 10, URL.replace("0123456789abcdef" * 2 + "01234567", "main"))})
        ),
        _catalog(entry(files={"model.onnx": FileEntry(SHA, 10, URL + "?download=1")})),
        # userinfo: the host is what follows the "@", so the reader of the catalogue would be misled
        _catalog(entry(files={"model.onnx": FileEntry(SHA, 10, URL.replace("huggingface.co", "hf.co@x.example"))})),
        _catalog(entry(files={"model.onnx": FileEntry(SHA, 10, URL.replace("huggingface.co", "u:p@huggingface.co"))})),
        json.dumps({"voices": {}}),
        json.dumps({"voices": [], "extra": 1}),
        json.dumps([]),
    ],
)
def test_bad_catalogue_rejected(text: str) -> None:
    with pytest.raises(IntegrityError):
        parse_catalog(text)
