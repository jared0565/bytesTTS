# SPDX-License-Identifier: MPL-2.0
import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from bytestts.catalog.details import Details, Person, VoiceDetails, parse_details, voice_label
from bytestts.catalog.manifest import FileEntry
from bytestts.errors import IntegrityError

SHA = "a" * 64
COMMIT = "0123456789abcdef" * 2 + "01234567"
CLIP = f"https://huggingface.co/fbmacagba/bytestts-en-vctk/resolve/{COMMIT}/samples/p228.wav"
BASE: dict[str, Any] = {
    "sentences": {"en": "Hello. This is how I sound reading your text aloud."},
    "voices": {
        "en-vctk": {
            "language_name": "English",
            "dataset": "VCTK",
            "speakers": {
                "p228": {
                    "gender": "Female",
                    "age": 22,
                    "accent": "English (England)",
                    "region": "Southern England",
                    "sample": {"sha256": SHA, "size": 140000, "url": CLIP},
                },
                "ED": {"note": "not in the VCTK speaker table"},
            },
        },
        "en-ljspeech": {"language_name": "English", "dataset": "LJSpeech", "reader": {"gender": "Female"}},
    },
}


def doc(change: Callable[[dict[str, Any]], object] = lambda d: None) -> bytes:
    """BASE, changed by `change` (which edits the copy in place), as JSON bytes."""
    d = copy.deepcopy(BASE)
    change(d)
    return json.dumps(d, ensure_ascii=False).encode("utf-8")


def vctk(d: dict[str, Any]) -> dict[str, Any]:
    voice: dict[str, Any] = d["voices"]["en-vctk"]
    return voice


def p228(d: dict[str, Any]) -> dict[str, Any]:
    person: dict[str, Any] = vctk(d)["speakers"]["p228"]
    return person


def sample(d: dict[str, Any]) -> dict[str, Any]:
    clip: dict[str, Any] = p228(d)["sample"]
    return clip


# --- the parser ------------------------------------------------------------------------------------------------


def test_parse_minimal_document() -> None:
    details = parse_details(doc())
    assert details.sentences == {"en": "Hello. This is how I sound reading your text aloud."}
    v = details.voice("en-vctk")
    assert v is not None and (v.language_name, v.dataset, v.reader) == ("English", "VCTK", None)
    assert v.speakers is not None and list(v.speakers) == ["p228", "ED"]
    assert v.person("p228") == Person(
        gender="Female",
        age=22,
        accent="English (England)",
        region="Southern England",
        sample=FileEntry(SHA, 140000, CLIP),
    )
    assert v.person("ED") == Person(note="not in the VCTK speaker table")
    assert v.person("p999") is None
    lj = details.voice("en-ljspeech")
    assert lj is not None and (lj.speakers, lj.reader) == (None, Person(gender="Female"))
    assert details.voice("tiny") is None


def _set(path: Callable[[dict[str, Any]], dict[str, Any]], key: str, value: object) -> Callable[[dict[str, Any]], None]:
    return lambda d: path(d).__setitem__(key, value)


def _drop(path: Callable[[dict[str, Any]], dict[str, Any]], key: str) -> Callable[[dict[str, Any]], None]:
    return lambda d: path(d).pop(key)


def _top(d: dict[str, Any]) -> dict[str, Any]:
    return d


def _ljspeech(d: dict[str, Any]) -> dict[str, Any]:
    voice: dict[str, Any] = d["voices"]["en-ljspeech"]
    return voice


@pytest.mark.parametrize(
    "data",
    [
        doc(_drop(_top, "voices")),
        doc(_set(_top, "extra", {})),
        doc(_set(vctk, "colour", "blue")),
        doc(_set(p228, "nickname", "x")),
        doc(_set(sample, "name", "p228.wav")),
        b'{"sentences": {"en": "a"}, "sentences": {"en": "b"}, "voices": {}}',  # a duplicate key
        doc() + b" " * (1 << 20),  # over 1 MiB
        b"\xff" + doc(),  # not UTF-8
        doc(_set(vctk, "reader", {})),  # both speakers and reader
        doc(_drop(_ljspeech, "reader")),  # neither
    ],
    ids=["no-voices", "top-extra", "voice-extra", "person-extra", "sample-extra", "duplicate", "big", "utf8", "both",
         "neither"],
)  # fmt: skip
def test_refuses_bad_shape(data: bytes) -> None:
    with pytest.raises(IntegrityError):
        parse_details(data)


@pytest.mark.parametrize(
    "change",
    [
        _set(p228, "gender", "F"),
        _set(p228, "age", 0),
        _set(p228, "age", 121),
        _set(p228, "age", True),
        _set(p228, "age", "22"),
        _set(p228, "accent", ""),
        _set(p228, "accent", "x" * 81),
        _set(p228, "accent", "a\tb"),
        _set(p228, "accent", "a\nb"),
        _set(p228, "accent", "a\x7fb"),
        _set(p228, "accent", "a b"),  # Unicode line separator (Zl)
        _set(p228, "accent", "a b"),  # paragraph separator (Zp)
        lambda d: d["sentences"].__setitem__("en-gb", "Hello."),
        lambda d: d["sentences"].__setitem__("en", "x" * 201),
    ],
    ids=["gender", "age-0", "age-121", "age-bool", "age-str", "accent-empty", "accent-81", "tab", "newline", "del",
         "line-sep", "para-sep", "sentence-key", "sentence-201"],
)  # fmt: skip
def test_refuses_bad_fields(change: Callable[[dict[str, Any]], None]) -> None:
    with pytest.raises(IntegrityError):
        parse_details(doc(change))


def test_accepts_the_edges() -> None:  # the limits above are inclusive at the other end
    def edges(d: dict[str, Any]) -> None:
        p228(d).update(age=120, accent="x" * 80, region="Stockton-on-tees")
        d["sentences"]["en"] = "x" * 200
        sample(d)["size"] = 1 << 20

    person = parse_details(doc(edges)).voices["en-vctk"].person("p228")
    assert person is not None and (person.age, person.accent) == (120, "x" * 80)
    assert parse_details(doc(_set(p228, "age", 1))).voices["en-vctk"].person("p228") is not None


@pytest.mark.parametrize(
    "change",
    [
        _drop(sample, "url"),
        _set(sample, "url", CLIP.replace("https://", "http://")),
        _set(sample, "url", CLIP.replace("huggingface.co", "a@huggingface.co")),
        _set(sample, "url", CLIP.replace(COMMIT, "main")),
        _set(sample, "url", CLIP.replace("/samples/p228.wav", "/samples/../x.wav")),
        _set(sample, "url", CLIP.replace("/samples/p228.wav", "/samples/CON.wav")),
        _set(sample, "url", CLIP.replace("/samples/p228.wav", "/model.onnx")),
        _set(sample, "size", 0),
        _set(sample, "size", (1 << 20) + 1),
        _set(sample, "sha256", "abc"),
    ],
    ids=["no-url", "http", "userinfo", "unpinned", "dotdot", "device", "not-a-sample", "size-0", "size-big", "sha"],
)
def test_refuses_bad_sample(change: Callable[[dict[str, Any]], None]) -> None:
    with pytest.raises(IntegrityError):
        parse_details(doc(change))


JUNK = [
    b"",
    b"[]",
    b"null",
    b"[" * 100000,
    b"{" * 100000,
    b"1",
    b'"x"',
    b"{}",
    b'{"sentences": [], "voices": {}}',
    b'{"sentences": {}, "voices": []}',
    b'{"sentences": {"en": 1}, "voices": {}}',
    b'{"sentences": {}, "voices": {"x": 3}}',
    b'{"sentences": {}, "voices": {"x": {"language_name": "a", "dataset": "b", "speakers": []}}}',
    b'{"sentences": {}, "voices": {"x": {"language_name": "a", "dataset": "b", "speakers": {"s": []}}}}',
    b'{"sentences": {}, "voices": {"x": {"language_name": "a", "dataset": "b", "reader": null}}}',
    b'{"sentences": {}, "voices": {"x": {"language_name": 1, "dataset": "b", "reader": {}}}}',
    b'{"sentences": {}, "voices": {"x": {"language_name": "a", "dataset": "b", "reader": {"age": 1.5}}}}',
    b'{"sentences": {}, "voices": {"x": {"language_name": "a", "dataset": "b", "reader": {"sample": []}}}}',
    b'{"sentences": {}, "voices": {"x": {"language_name": "a", "dataset": "b", "reader": {"sample": {"sha256": 1}}}}}',
    doc(_set(sample, "size", "140000")),
]


@pytest.mark.parametrize("data", JUNK)
def test_every_failure_is_integrity_error(data: bytes) -> None:
    with pytest.raises(IntegrityError):
        parse_details(data)


@pytest.mark.parametrize("data", ["{}", None, 3, ["x"]])
def test_non_bytes_is_integrity_error(data: Any) -> None:  # the drain's fuzzer calls it with any argument
    with pytest.raises(IntegrityError):
        parse_details(data)


# --- labels ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "person", "label"),
    [
        ("p228", Person("Female", 22, "English (England)", "Southern England"),
         "p228 · Female, 22 · English (England), Southern England"),
        ("p248", Person("Female", 23, "Indian"), "p248 · Female, 23 · Indian"),
        ("p280", Person("Female", 25, region="France"), "p280 · Female, 25 · France"),
        ("ED", Person(note="not in the VCTK speaker table"), "ED · not in the VCTK speaker table"),
        ("p999", Person(), "p999"),
    ],
)  # fmt: skip
def test_speaker_labels(name: str, person: Person, label: str) -> None:
    assert person.label(name) == label
    assert person.describe() == label.removeprefix(name).removeprefix(" · ")


ACCENTS = ["English (England)", "American", "Scottish", "Irish", "Canadian", "Northern Irish", "South African",
           "Indian", "Australian", "New Zealand", "Welsh"]  # fmt: skip


def _vctk_like(accents: list[str]) -> tuple[list[str], VoiceDetails]:
    people = {f"p{300 + i}": Person("Female", 30, accents[i % len(accents)]) for i in range(108)}
    names = ["ED", *people]
    return names, VoiceDetails("English", "VCTK", {**people, "ED": Person(note="x")}, None)


def test_voice_labels() -> None:
    names, d = _vctk_like(ACCENTS)
    assert len(names) == 109
    assert voice_label("en-vctk", names, d) == "English · VCTK · 109 speakers, 11 accents"
    names, d = _vctk_like(["Scottish"])
    assert voice_label("en-vctk", names, d) == "English · VCTK · 109 speakers"
    es = VoiceDetails("Spanish", "CSS10", {"tux": Person()}, None)
    assert voice_label("es-css10", ["tux"], es) == "Spanish · CSS10 · 1 speaker"
    lj = VoiceDetails("English", "LJSpeech", None, Person())
    assert voice_label("en-ljspeech", [], lj) == "English · LJSpeech · 1 speaker"  # an empty table is 1 speaker
    assert voice_label("tiny", [], None) == "tiny"


def test_sentences_and_reader_label() -> None:
    lj = VoiceDetails("English", "LJSpeech", None, Person())
    details = Details({"en": "Hello.", "es": "Hola."}, {"en-ljspeech": lj})
    assert details.sentence("en-gb") == "Hello."
    assert details.sentence("es") == "Hola."
    assert details.sentence("de") is None
    assert lj.reader_label() == "LJSpeech"
    assert VoiceDetails("English", "LJSpeech", None, Person("Female")).reader_label() == "LJSpeech · Female"
    assert VoiceDetails("Spanish", "CSS10", {"tux": Person()}, None).reader_label() is None
