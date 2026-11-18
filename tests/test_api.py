# SPDX-License-Identifier: MPL-2.0
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from helpers import TEST_THREADS, TINY, edit_json, repin

import bytestts
from bytestts import Audio, BytesTTS, VoiceInfo
from bytestts.catalog import catalog
from bytestts.catalog.cache import voice_dir
from bytestts.catalog.manifest import FileEntry, VoiceManifest
from bytestts.engines.vits_onnx import ENGINES, VitsOnnxEngine
from bytestts.errors import (
    BytesTTSError,
    ConsentMissing,
    FrontendUnavailable,
    IntegrityError,
    NoSpeakableText,
    NotSupported,
    VoiceNotFound,
)

TINY_SR = 22050


@pytest.fixture(scope="module")
def tts() -> BytesTTS:
    return BytesTTS.load_path(TINY, device="cpu", threads=TEST_THREADS)


def copy_tree(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return dest


def load(d: Path) -> BytesTTS:
    return BytesTTS.load_path(d, device="cpu", threads=TEST_THREADS)


# --- synthesis ---


def test_synthesize_returns_audio(tts: BytesTTS) -> None:
    a = tts.synthesize("Hello. World.")
    assert isinstance(a, Audio) and a.sample_rate == TINY_SR and a.duration > BytesTTS.SENTENCE_PAUSE_S


def test_stream_yields_per_sentence(tts: BytesTTS) -> None:
    parts = list(tts.stream("One. Two. Three."))  # capitalised so pysbd splits
    assert len(parts) == 3 and all(isinstance(a, Audio) and a.sample_rate == TINY_SR for a in parts)


def test_synthesize_is_the_stream_joined_by_pauses(tts: BytesTTS) -> None:
    # lengths are deterministic: the tiny voice's duration predictor is not stochastic
    parts = [len(a.samples) for a in tts.stream("One. Two. Three.")]
    whole = tts.synthesize("One. Two. Three.").samples
    gap = round(BytesTTS.SENTENCE_PAUSE_S * TINY_SR)
    assert len(whole) == sum(parts) + 2 * gap
    assert not whole[parts[0] : parts[0] + gap].any()  # the pause is silence


def test_empty_text(tts: BytesTTS) -> None:
    with pytest.raises(NoSpeakableText):
        tts.synthesize("  ")


def test_stream_checks_its_arguments_before_iterating(tts: BytesTTS) -> None:
    with pytest.raises(NoSpeakableText):
        tts.stream("  ")
    with pytest.raises(ValueError):
        tts.stream("Hi.", speed=3.0)
    with pytest.raises(VoiceNotFound):
        tts.stream("Hi.", speaker="bob")  # the tiny voice has one speaker


def test_faster_is_shorter(tts: BytesTTS) -> None:
    text = "The quick brown fox."
    assert len(tts.synthesize(text, speed=2.0).samples) < len(tts.synthesize(text).samples)


def test_info_and_card(tts: BytesTTS) -> None:
    manifest = VoiceManifest.from_json(TINY / "voice.json")
    assert tts.info == VoiceInfo.from_manifest(manifest)
    assert tts.card is None


@pytest.mark.parametrize("threads", [3, None])
def test_threads_and_device_reach_the_engine(monkeypatch: pytest.MonkeyPatch, threads: int | None) -> None:
    seen: dict[str, Any] = {}

    def spy(*args: Any, **kwargs: Any) -> VitsOnnxEngine:
        seen.update(kwargs)  # what the API asked for; the session itself stays single-threaded (conftest)
        return VitsOnnxEngine(*args, **{**kwargs, "threads": TEST_THREADS})

    monkeypatch.setitem(ENGINES, "vits-onnx", spy)
    BytesTTS.load_path(TINY, device="cpu", threads=threads)
    assert (seen["device"], seen["threads"]) == ("cpu", threads)


# --- loading ---


def test_path_with_spaces(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path / "my vöices" / "tiny")
    assert load(d).info.id == "tiny" and load(d).synthesize("Hi.").duration > 0


def test_load_unknown_voice() -> None:
    with pytest.raises(VoiceNotFound):
        BytesTTS.load("nope", offline=True)


def test_load_from_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BYTESTTS_HOME", str(tmp_path / "ä cache"))
    m = VoiceManifest.from_json(TINY / "voice.json")
    files = {name: FileEntry(e.sha256, e.size, f"https://example.com/{name}") for name, e in m.files.items()}
    m = VoiceManifest(**{**m.__dict__, "files": files})
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"tiny": m})
    d = voice_dir(m)
    d.mkdir(parents=True)
    for name in m.files:
        shutil.copy(TINY / name, d / name)
    tts = BytesTTS.load("tiny", offline=True, device="cpu", threads=TEST_THREADS)
    assert tts.info.id == "tiny" and tts.synthesize("Hi.").duration > 0


def test_tampered_package_refused(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path / "v")
    data = bytearray((d / "model.onnx").read_bytes())
    data[200] ^= 1
    (d / "model.onnx").write_bytes(bytes(data))
    with pytest.raises(IntegrityError, match=r"model\.onnx"):
        load(d)


def test_unknown_engine(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path / "v")
    edit_json(d / "voice.json", lambda m: m.__setitem__("engine", "other"))
    with pytest.raises(NotSupported, match="other"):
        load(d)


def test_sample_rates_must_agree(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path / "v")
    edit_json(d / "config.json", lambda c: c.__setitem__("sample_rate", 16000))
    repin(d, "config.json")
    with pytest.raises(IntegrityError, match="sample_rate"):
        load(d)


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.pop("use_phonemes"),  # a missing key
        lambda c: c.__setitem__("characters", "abc"),  # a wrong type
        lambda c: c.update(use_phonemes=True, phoneme_language=None),  # a phoneme voice with no phoneme language
        lambda c: c["characters"].update(punctuations="!a"),  # "a" twice in the vocabulary, found by the frontend
    ],
    ids=["missing-key", "wrong-type", "no-phoneme-language", "duplicate-symbol"],
)
def test_bad_text_settings_are_an_integrity_error(tmp_path: Path, edit: Any) -> None:
    # a pinned config.json can still be malformed: the error must be BytesTTS's (CLI exit 3), not KeyError & co.
    d = copy_tree(TINY, tmp_path / "v")
    edit_json(d / "config.json", edit)
    repin(d, "config.json")
    with pytest.raises(IntegrityError, match=r"config\.json"):
        load(d)


def test_speakers_for_a_single_speaker_model_refused(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path / "v")  # the tiny model takes no speaker id
    (d / "speakers.json").write_bytes(json.dumps({"a": 0, "b": 1}).encode())
    repin(d, "speakers.json")
    edit_json(d / "voice.json", lambda m: m.__setitem__("speakers", ["a", "b"]))
    with pytest.raises(IntegrityError, match="single speaker"):
        load(d)


def test_fine_tuned_voice_exposes_its_card(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path / "v")
    card = {"consent": {"statement": "I, A. Speaker, consent."}, "speaker": "A. Speaker"}
    (d / "card.json").write_bytes(json.dumps(card).encode())
    repin(d, "card.json")
    edit_json(d / "voice.json", lambda m: m.__setitem__("origin", "fine-tuned"))
    assert load(d).card == card


def test_fine_tuned_voice_without_consent_refused(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path / "v")
    edit_json(d / "voice.json", lambda m: m.__setitem__("origin", "fine-tuned"))
    with pytest.raises(ConsentMissing):
        load(d)


@pytest.mark.parametrize("card", [b"[1, 2]", b'"a card"', b"null"])
def test_a_card_must_be_a_json_object(tmp_path: Path, card: bytes) -> None:  # a catalogue voice's card too
    d = copy_tree(TINY, tmp_path / "v")
    (d / "card.json").write_bytes(card)
    repin(d, "card.json")
    with pytest.raises(IntegrityError, match=r"card\.json must be a JSON object"):
        load(d)


_RECORDING: list[tuple[str, list[str]]] = []  # (directory, sink) while a test records opens


def _record_opens(event: str, args: tuple[Any, ...]) -> None:  # an audit hook: io.open and os.open both raise it
    if event == "open" and _RECORDING and isinstance(args[0], (str, bytes, os.PathLike)):
        directory, sink = _RECORDING[-1]
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if os.path.dirname(path) == directory:
            sink.append(os.path.basename(path))


sys.addaudithook(_record_opens)  # audit hooks cannot be removed; it is inert while _RECORDING is empty


def test_each_package_file_is_read_once(tmp_path: Path) -> None:
    """What is used is what was hashed: a file opened again after its check could have been swapped in between."""
    d = copy_tree(TINY, tmp_path / "v")
    (d / "card.json").write_bytes(b'{"speaker": "LJ"}')
    repin(d, "card.json")
    opened: list[str] = []
    _RECORDING.append((os.path.normcase(os.path.abspath(d)), opened))
    try:
        load(d)
    finally:
        _RECORDING.clear()
    listed = json.loads((d / "voice.json").read_bytes())["files"]
    assert sorted(opened) == sorted(["voice.json", *listed])


def test_listed_speakers_need_speakers_json_pinned(tmp_path: Path) -> None:
    d = copy_tree(TINY, tmp_path / "v")
    (d / "speakers.json").write_bytes(json.dumps({"a": 0, "b": 1}).encode())  # valid, but not pinned: never read
    edit_json(d / "voice.json", lambda m: m.__setitem__("speakers", ["a", "b"]))
    with pytest.raises(IntegrityError, match=r"speakers\.json"):
        load(d)


@pytest.mark.parametrize(
    "ids",
    [
        {"a": 0, "c": 1},  # names differ from the manifest's
        {"a": 0},
        {"a": 0, "b": -1},
        {"a": 0, "b": True},
        {"a": 0, "b": 0},  # two names, one id
        ["a", "b"],
    ],
)
def test_speakers_json_must_match_the_manifest(tmp_path: Path, ids: Any) -> None:
    d = copy_tree(TINY, tmp_path / "v")
    (d / "speakers.json").write_bytes(json.dumps(ids).encode())
    repin(d, "speakers.json")
    edit_json(d / "voice.json", lambda m: m.__setitem__("speakers", ["a", "b"]))
    with pytest.raises(IntegrityError, match=r"speakers\.json"):
        load(d)


# --- the public surface ---


def test_public_exports() -> None:
    names = {"BytesTTS", "list_voices", "Audio", "VoiceInfo", "BytesTTSError", "VoiceNotFound", "IntegrityError"}
    names |= {"FrontendUnavailable", "ConsentMissing", "NotSupported", "NoSpeakableText"}
    assert names <= set(bytestts.__all__)
    for name in names:
        assert getattr(bytestts, name) is not None
    assert bytestts.ConsentMissing is ConsentMissing and bytestts.FrontendUnavailable is FrontendUnavailable
    assert issubclass(bytestts.NotSupported, BytesTTSError)
    assert isinstance(bytestts.list_voices(), list)
    assert np is not None
