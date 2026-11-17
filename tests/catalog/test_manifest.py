# SPDX-License-Identifier: MPL-2.0
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from helpers import TINY

from bytestts.catalog.manifest import (
    MAX_JSON_BYTES,
    FileEntry,
    VoiceManifest,
    safe_name,
    sha256_file,
    verify_package,
)
from bytestts.errors import ConsentMissing, IntegrityError


def copy_tree(src: Path, tmp_path: Path) -> Path:
    d = tmp_path / "voice"
    shutil.copytree(src, d)
    return d


def edit_manifest(d: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = json.loads((d / "voice.json").read_bytes())
    change(data)
    (d / "voice.json").write_bytes(json.dumps(data).encode())


def pin(d: Path, name: str, content: bytes) -> None:
    (d / name).write_bytes(content)
    entry = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    edit_manifest(d, lambda m: m["files"].__setitem__(name, entry))


def set_origin(d: Path, origin: str) -> None:
    edit_manifest(d, lambda m: m.__setitem__("origin", origin))


# --- verify_package ---


def test_verify_ok() -> None:
    m = verify_package(TINY)
    assert (m.id, m.engine, m.origin, m.sample_rate) == ("tiny", "vits-onnx", "catalog", 22050)
    assert set(m.files) == {"model.onnx", "config.json"} and isinstance(m.files["model.onnx"], FileEntry)


def test_tamper_detected(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path)
    data = bytearray((d / "model.onnx").read_bytes())
    data[100] ^= 1  # same size, different bytes: only the hash can tell
    (d / "model.onnx").write_bytes(bytes(data))
    with pytest.raises(IntegrityError, match=r"model\.onnx"):
        verify_package(d)


def test_size_mismatch_detected(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path)
    (d / "config.json").write_bytes((d / "config.json").read_bytes() + b" ")
    with pytest.raises(IntegrityError, match=r"config\.json.*size"):
        verify_package(d)


def test_missing_file_detected(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path)
    (d / "model.onnx").unlink()
    with pytest.raises(IntegrityError, match=r"model\.onnx"):
        verify_package(d)


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match=r"voice\.json"):
        verify_package(tmp_path)


@pytest.mark.parametrize("name", ["model.onnx", "config.json"])
def test_engine_files_must_be_listed(tmp_path: Path, name: str) -> None:  # else the loader reads an unpinned file
    d = copy_tree(TINY, tmp_path)
    edit_manifest(d, lambda m: m["files"].pop(name))
    with pytest.raises(IntegrityError, match=name):
        verify_package(d)


def test_symlinked_file_refused(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path)
    real = tmp_path / "elsewhere.onnx"
    shutil.move(d / "model.onnx", real)
    try:
        os.symlink(real, d / "model.onnx")
    except OSError:  # Windows without Developer Mode
        pytest.skip("cannot create symlinks here")
    with pytest.raises(IntegrityError, match=r"model\.onnx"):
        verify_package(d)


def test_unsafe_key_rejected_before_any_file_is_read(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path)
    (tmp_path / "secret.bin").write_bytes(b"x")
    pinned = {"sha256": hashlib.sha256(b"x").hexdigest(), "size": 1}
    edit_manifest(d, lambda m: m["files"].__setitem__("../secret.bin", pinned))
    with pytest.raises(IntegrityError, match="unsafe"):
        verify_package(d)


# --- consent ---


def test_finetuned_without_consent(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path)
    set_origin(d, "fine-tuned")
    with pytest.raises(ConsentMissing):
        verify_package(d)


def test_finetuned_with_consent(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path)
    pin(d, "card.json", json.dumps({"consent": {"statement": "I, A. Speaker, consent."}}).encode())
    set_origin(d, "fine-tuned")
    assert verify_package(d).origin == "fine-tuned"


def test_finetuned_card_must_be_pinned(tmp_path: Path) -> None:  # an unpinned card could be swapped freely
    d = copy_tree(TINY, tmp_path)
    (d / "card.json").write_bytes(json.dumps({"consent": {"statement": "I consent."}}).encode())
    set_origin(d, "fine-tuned")
    with pytest.raises(ConsentMissing, match=r"card\.json"):
        verify_package(d)


@pytest.mark.parametrize(
    "card", [b"{}", b'{"consent": {}}', b'{"consent": {"statement": "  "}}', b'{"consent": "yes"}', b"[1]", b"{"]
)
def test_finetuned_card_needs_a_statement(tmp_path: Path, card: bytes) -> None:
    d = copy_tree(TINY, tmp_path)
    pin(d, "card.json", card)
    set_origin(d, "fine-tuned")
    with pytest.raises(ConsentMissing):
        verify_package(d)


# --- safe_name ---


@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "a/b",
        "a\\b",
        "C:x",
        "..",
        ".",
        "",
        "a b",
        "model.onnx.",
        "CON",
        "nul.onnx",
        "Aux",
        "com1.json",
        "LPT9",
        "x" * 129,
        "café",
    ],
)
def test_unsafe_file_names_rejected(name: str) -> None:
    with pytest.raises(IntegrityError):
        safe_name(name)


@pytest.mark.parametrize("name", ["model.onnx", "config.json", "en-ljspeech", "a_b-1.x", ".hidden", "console.json"])
def test_safe_names_accepted(name: str) -> None:
    assert safe_name(name) == name


# --- manifest parsing ---


def _good() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((TINY / "voice.json").read_bytes())
    return data


def _write(tmp_path: Path, data: Any) -> Path:
    p = tmp_path / "voice.json"
    p.write_bytes(json.dumps(data).encode())
    return p


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.pop("licence"),  # missing key
        lambda m: m.__setitem__("extra", 1),  # unknown key
        lambda m: m.__setitem__("origin", "downloaded"),
        lambda m: m.__setitem__("sample_rate", 0),
        lambda m: m.__setitem__("sample_rate", True),  # bool is an int to Python, not to us
        lambda m: m.__setitem__("sample_rate", 22050.0),
        lambda m: m.__setitem__("id", "../x"),  # the id becomes a cache directory (Task 13)
        lambda m: m.__setitem__("language", "english"),
        lambda m: m.__setitem__("attribution", ""),
        lambda m: m["files"]["model.onnx"].__setitem__("sha256", "AB" * 32),  # uppercase
        lambda m: m["files"]["model.onnx"].__setitem__("sha256", "ab" * 31),
        lambda m: m["files"]["model.onnx"].__setitem__("size", -1),
        lambda m: m["files"]["model.onnx"].__setitem__("size", False),
        lambda m: m["files"]["model.onnx"].__setitem__("url", 5),
        lambda m: m["files"]["model.onnx"].__setitem__("mode", "x"),
        lambda m: m.__setitem__("files", []),
        lambda m: m.__setitem__("speakers", ["a", "a"]),
        lambda m: m.__setitem__("featured", ["b"]),  # featured must be a listed speaker
        lambda m: m.__setitem__("speakers", "abc"),
    ],
)
def test_bad_manifest_rejected(tmp_path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = _good()
    change(data)
    with pytest.raises(IntegrityError):
        VoiceManifest.from_json(_write(tmp_path, data))


@pytest.mark.parametrize(
    "corrupt",
    [
        # a duplicate key json would resolve silently to its last value, which here is the valid one
        lambda b: b.replace(b"{", b'{"licence": "hidden", ', 1),
        lambda b: b.replace(b'"sample_rate": 22050', b'"sample_rate": NaN'),
        lambda b: b"[" + b + b"]",
        lambda b: bytes([0xFF, 0xFE]) + b,  # not UTF-8
        lambda b: b[:-5],
    ],
)
def test_malformed_json_rejected(tmp_path: Path, corrupt: Callable[[bytes], bytes]) -> None:
    good = (TINY / "voice.json").read_bytes()
    p = tmp_path / "voice.json"
    p.write_bytes(corrupt(good))
    with pytest.raises(IntegrityError):
        VoiceManifest.from_json(p)


def test_duplicate_key_is_the_reason() -> None:  # the corrupted manifest above is otherwise valid
    good = (TINY / "voice.json").read_bytes()
    assert VoiceManifest.from_dict(json.loads(good.replace(b"{", b'{"licence": "hidden", ', 1))).licence == "MPL-2.0"


def test_files_cannot_be_changed_after_validation() -> None:  # so every file name stays safe
    m = VoiceManifest.from_json(TINY / "voice.json")
    with pytest.raises(TypeError):
        m.files["../x"] = m.files["model.onnx"]  # type: ignore[index]


def test_oversized_manifest_rejected(tmp_path: Path) -> None:
    p = tmp_path / "voice.json"
    p.write_bytes(b" " * (MAX_JSON_BYTES + 1))
    with pytest.raises(IntegrityError, match="too large"):
        VoiceManifest.from_json(p)


def test_manifest_round_trips() -> None:
    m = VoiceManifest.from_json(TINY / "voice.json")
    assert VoiceManifest.from_dict(m.to_dict()) == m
    assert m.to_dict() == _good()  # the fixture omits url: None stays omitted


def test_manifest_digest_is_canonical(tmp_path: Path) -> None:
    m = VoiceManifest.from_json(TINY / "voice.json")
    reordered = tmp_path / "voice.json"
    reordered.write_bytes(json.dumps(dict(reversed(list(_good().items()))), indent=7).encode())
    assert VoiceManifest.from_json(reordered).manifest_digest() == m.manifest_digest()
    assert len(m.manifest_digest()) == 64

    data = _good()
    data["files"]["model.onnx"]["sha256"] = "0" * 64
    assert VoiceManifest.from_json(_write(tmp_path, data)).manifest_digest() != m.manifest_digest()


def test_sha256_file_across_chunks(tmp_path: Path) -> None:
    data = os.urandom(3 * 1024 * 1024 + 17)  # spans several 1 MiB reads
    p = tmp_path / "blob"
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
