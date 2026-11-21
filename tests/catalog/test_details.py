# SPDX-License-Identifier: MPL-2.0
import copy
import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from gui_helpers import BytesServer, wait_until

from bytestts.catalog import details
from bytestts.catalog.cache import cache_root
from bytestts.catalog.catalog import load_catalog
from bytestts.catalog.details import (
    Details,
    Person,
    VoiceDetails,
    load_details,
    parse_details,
    voice_label,
    voice_name,
)
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


def test_voice_name() -> None:  # status lines and recording names: the label without its counts
    found = load_details()
    assert voice_name("en-vctk", found.voice("en-vctk")) == "English · VCTK"
    assert voice_name("en-ljspeech", found.voice("en-ljspeech")) == "English · LJSpeech"
    assert voice_name("tiny", None) == "tiny"  # no details: the id


def test_sentences_and_reader_label() -> None:
    lj = VoiceDetails("English", "LJSpeech", None, Person())
    details = Details({"en": "Hello.", "es": "Hola."}, {"en-ljspeech": lj})
    assert details.sentence("en-gb") == "Hello."
    assert details.sentence("es") == "Hola."
    assert details.sentence("de") is None
    assert lj.reader_label() == "LJSpeech"
    assert VoiceDetails("English", "LJSpeech", None, Person("Female")).reader_label() == "LJSpeech · Female"
    assert VoiceDetails("Spanish", "CSS10", {"tux": Person()}, None).reader_label() is None


# --- the shipped details.json (§7.1: things that must never break) ---------------------------------------------


def test_details_cover_every_catalogue_voice() -> None:
    assert set(load_details().voices) == set(load_catalog())


def test_speaker_tables_are_described_exactly() -> None:  # es-css10's and fr-css10's one-speaker tables included
    details = load_details()
    for voice_id, m in load_catalog().items():
        v = details.voice(voice_id)
        assert v is not None
        if m.speakers:
            assert v.speakers is not None and list(v.speakers) == m.speakers, voice_id


def test_reader_only_for_empty_tables() -> None:  # today only en-ljspeech
    details = load_details()
    readers = {voice_id for voice_id, v in details.voices.items() if v.reader is not None}
    assert readers == {voice_id for voice_id, m in load_catalog().items() if not m.speakers} == {"en-ljspeech"}


def test_every_language_has_a_sentence() -> None:
    details = load_details()
    assert all(details.sentence(m.language) for m in load_catalog().values())


def test_samples_live_in_their_own_voice_repo() -> None:  # vacuous until the clips are pinned; it guards the pins
    for voice_id, v in load_details().voices.items():
        people = [*(v.speakers or {}).values(), *([v.reader] if v.reader else [])]
        for p in people:
            if p.sample is not None:
                assert p.sample.url is not None
                assert p.sample.url.startswith(f"https://huggingface.co/fbmacagba/bytestts-{voice_id}/"), voice_id


def test_every_speaker_and_reader_has_a_sample() -> None:  # design §7.1: every Sample and Preview can play a clip
    found = load_details()
    pinned = 0
    for voice_id, manifest in load_catalog().items():
        v = found.voice(voice_id)
        assert v is not None, voice_id
        people = [v.person(name) for name in manifest.speakers] if manifest.speakers else [v.reader]
        for person in people:
            assert person is not None and person.sample is not None, voice_id
            pinned += 1
    assert pinned == 112


def test_vctk_counts() -> None:  # the design's §6.2 table, counted from the shipped file
    v = load_details().voice("en-vctk")
    assert v is not None and v.speakers is not None
    people = list(v.speakers.values())
    assert Counter(p.gender for p in people) == {"Female": 62, "Male": 46, None: 1}
    assert Counter(p.accent for p in people) == {
        "English (England)": 33, "American": 21, "Scottish": 19, "Irish": 9, "Canadian": 8, "Northern Irish": 6,
        "South African": 4, "Indian": 3, "Australian": 2, "New Zealand": 1, "Welsh": 1, None: 2,
    }  # fmt: skip
    assert sorted(name for name, p in v.speakers.items() if p.accent is None) == ["ED", "p280"]
    assert voice_label("en-vctk", load_catalog()["en-vctk"].speakers, v) == "English · VCTK · 109 speakers, 11 accents"
    ed = v.speakers["ED"]  # its facts; its sample is pinned like every other speaker's
    assert (ed.gender, ed.age, ed.accent, ed.region, ed.note) == (
        None,
        None,
        None,
        None,
        "not in the VCTK speaker table",
    )
    assert ed.sample is not None
    assert v.speakers["p228"].label("p228") == "p228 · Female, 22 · English (England), Southern England"


# --- sample_bytes: fetch, cache and check a clip ----------------------------------------------------------------

CLIP_BYTES = b"RIFF" + bytes(range(256)) * 20
OTHER_BYTES = b"RIFF" + bytes(reversed(range(256))) * 20  # the same length


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BYTESTTS_HOME", str(tmp_path))
    return tmp_path


def _entry(server: BytesServer, path: str, data: bytes = CLIP_BYTES) -> FileEntry:
    """An http entry on the local server, built directly: the parser's https rule doesn't apply here."""
    return FileEntry(hashlib.sha256(data).hexdigest(), len(data), f"{server.url_base}/{path}")


def _cached(entry: FileEntry) -> Path:
    return cache_root() / "samples" / f"{entry.sha256}.wav"


def test_first_call_fetches_and_caches(home: Path) -> None:
    with BytesServer({"a.wav": CLIP_BYTES}) as server:
        entry = _entry(server, "a.wav")
        assert details.sample_bytes(entry) == CLIP_BYTES
    assert _cached(entry).read_bytes() == CLIP_BYTES


def test_second_call_makes_no_request(home: Path) -> None:
    with BytesServer({"a.wav": CLIP_BYTES}) as server:
        entry = _entry(server, "a.wav")
        details.sample_bytes(entry)
        assert details.sample_bytes(entry) == CLIP_BYTES
        assert server.hits["/a.wav"] == 1


def test_same_clip_at_once_is_fetched_once(home: Path) -> None:
    with BytesServer({"a.wav": CLIP_BYTES}) as server:
        entry = _entry(server, "stall/a.wav")
        results: list[bytes] = []
        threads = [threading.Thread(target=lambda: results.append(details.sample_bytes(entry))) for _ in range(2)]
        for t in threads:
            t.start()
        wait_until(lambda: server.hits["/stall/a.wav"] == 1)
        time.sleep(0.2)  # the second call is waiting for the first by now
        server.release.set()
        for t in threads:
            t.join(10)
        assert results == [CLIP_BYTES, CLIP_BYTES]
        assert server.hits["/stall/a.wav"] == 1


def test_a_stalled_clip_does_not_hold_up_another(home: Path) -> None:
    with BytesServer({"a.wav": CLIP_BYTES, "b.wav": OTHER_BYTES}) as server:
        stalled = threading.Thread(target=details.sample_bytes, args=(_entry(server, "stall/a.wav"),), daemon=True)
        stalled.start()
        wait_until(lambda: server.hits["/stall/a.wav"] == 1)
        start = time.monotonic()
        assert details.sample_bytes(_entry(server, "b.wav", OTHER_BYTES)) == OTHER_BYTES
        assert time.monotonic() - start < 2
        server.release.set()
        stalled.join(10)


def test_a_stalled_clip_gives_up_within_the_budget(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The server stops working on a clip before the page gives up on it (30 s), so a stalled fetch never holds a
    worker after its request has ended."""
    monkeypatch.setattr(details, "SAMPLE_BUDGET_S", 0.5)  # 25 s in use
    monkeypatch.setattr(details, "SAMPLE_TIMEOUT", (10.0, 5.0))  # without the budget, the 5 s read limit governs
    with BytesServer({"a.wav": CLIP_BYTES}) as server:
        start = time.monotonic()
        # exactly "timed out", the page's own wording: not requests' "HTTPConnectionPool(...): Read timed out."
        with pytest.raises(TimeoutError, match=r"^timed out$"):
            details.sample_bytes(_entry(server, "stall/a.wav"))
        assert time.monotonic() - start < 2


def test_waiting_for_a_stalled_clip_gives_up(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with BytesServer({"a.wav": CLIP_BYTES}) as server:
        entry = _entry(server, "stall/a.wav")
        first = threading.Thread(target=details.sample_bytes, args=(entry,), daemon=True)
        first.start()  # with the 25 s budget: it holds the clip's lock until the server lets go
        wait_until(lambda: server.hits["/stall/a.wav"] == 1)
        monkeypatch.setattr(details, "SAMPLE_BUDGET_S", 0.2)  # the second request's budget, spent waiting
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            details.sample_bytes(entry)
        assert time.monotonic() - start < 1
        server.release.set()
        first.join(10)


def test_wrong_bytes_are_refused(home: Path) -> None:
    with BytesServer({"a.wav": OTHER_BYTES}) as server, pytest.raises(IntegrityError):
        details.sample_bytes(_entry(server, "a.wav"))  # pinned to CLIP_BYTES
    assert not (home / "samples").exists() or not any((home / "samples").iterdir())


def test_oversize_announcement_is_cut_off(home: Path) -> None:
    with BytesServer({"a.wav": CLIP_BYTES + b"x" * 100}) as server, pytest.raises(IntegrityError, match="announces"):
        details.sample_bytes(_entry(server, "a.wav"))
    assert not (home / "samples").exists() or not any((home / "samples").iterdir())


def test_tampered_cache_is_fetched_again(home: Path) -> None:
    with BytesServer({"a.wav": CLIP_BYTES}) as server:
        entry = _entry(server, "a.wav")
        details.sample_bytes(entry)
        _cached(entry).write_bytes(OTHER_BYTES)
        assert details.sample_bytes(entry) == CLIP_BYTES
        assert server.hits["/a.wav"] == 2


def test_a_clip_with_a_bad_hash_is_refused(home: Path) -> None:  # the hash names the cache file
    with BytesServer({"a.wav": CLIP_BYTES}) as server:
        for sha256 in ("../" + "a" * 61, "A" * 64, ""):
            with pytest.raises(IntegrityError):
                details.sample_bytes(FileEntry(sha256, len(CLIP_BYTES), f"{server.url_base}/a.wav"))
        assert server.hits["/a.wav"] == 0
