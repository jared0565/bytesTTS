# SPDX-License-Identifier: MPL-2.0
"""The public API: load a voice, turn text into audio."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from bytestts.audio import Audio, concat
from bytestts.catalog.catalog import VoiceInfo
from bytestts.catalog.download import ensure_voice
from bytestts.catalog.manifest import CARD, VoiceManifest, read_json, verify_package
from bytestts.engines.base import SpeechEngine, SynthesisOptions, VoiceRef
from bytestts.engines.vits_onnx import ENGINES, VitsDefaults
from bytestts.errors import IntegrityError, NotSupported, VoiceNotFound
from bytestts.text.frontend import FrontendConfig, TextFrontend

CONFIG = "config.json"
SPEAKERS = "speakers.json"  # {speaker name: model speaker id}, written by the converter (Task 16)

Device = Literal["auto", "cpu", "cuda"]


def _speaker_ids(directory: Path, manifest: VoiceManifest) -> dict[str, int] | None:
    """Speaker name -> model id, from the pinned speakers.json; featured speakers first, so the default is one."""
    if not manifest.speakers:
        return None
    if SPEAKERS not in manifest.files:
        raise IntegrityError(f"a voice that lists speakers must also list {SPEAKERS} in voice.json")
    ids = read_json(directory / SPEAKERS, SPEAKERS)
    if not isinstance(ids, dict) or set(ids) != set(manifest.speakers):
        raise IntegrityError(f"{SPEAKERS} must map exactly the speakers voice.json lists")
    values = list(ids.values())
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values) or len(set(values)) != len(values):
        raise IntegrityError(f"{SPEAKERS} must give each speaker its own integer id >= 0")
    order = [*manifest.featured, *(s for s in manifest.speakers if s not in manifest.featured)]
    return {name: ids[name] for name in order}


def _config(directory: Path, manifest: VoiceManifest) -> tuple[dict[str, Any], VitsDefaults]:
    config = read_json(directory / CONFIG, CONFIG)
    if not isinstance(config, dict):
        raise IntegrityError(f"{CONFIG} must be a JSON object")
    try:
        defaults = VitsDefaults.from_config(config)
    except (KeyError, TypeError, ValueError) as e:
        raise IntegrityError(f"{CONFIG}: bad or missing synthesis settings ({e})") from None
    if defaults.sample_rate != manifest.sample_rate:
        raise IntegrityError(
            f"{CONFIG} sample_rate {defaults.sample_rate} differs from voice.json's {manifest.sample_rate}"
        )
    return config, defaults


def _frontend(config: Mapping[str, Any]) -> TextFrontend:
    """The voice's text frontend. A malformed config.json is the package's fault: IntegrityError, not KeyError & co.
    NotSupported (an unknown cleaner or symbol layout) and FrontendUnavailable (no espeak-ng) pass unchanged."""
    try:
        return TextFrontend(FrontendConfig.from_voice_config(config))
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise IntegrityError(f"{CONFIG}: bad or missing text settings ({type(e).__name__}: {e})") from None


class BytesTTS:
    """A loaded voice. Make one with `load` (a catalogue voice) or `load_path` (a voice package directory)."""

    SENTENCE_PAUSE_S = 0.45

    def __init__(
        self, *, info: VoiceInfo, card: Mapping[str, Any] | None, frontend: TextFrontend, engine: SpeechEngine
    ) -> None:
        self.info = info
        self.card = card
        self._frontend = frontend
        self._engine = engine

    @classmethod
    def load(
        cls, voice_id: str, *, device: Device = "auto", offline: bool = False, threads: int | None = None
    ) -> BytesTTS:
        """A catalogue voice, downloaded and verified on first use (`offline` never downloads)."""
        return cls.load_path(ensure_voice(voice_id, offline=offline), device=device, threads=threads)

    @classmethod
    def load_path(
        cls, path: str | os.PathLike[str], *, device: Device = "auto", threads: int | None = None
    ) -> BytesTTS:
        """A voice package directory: every file is verified against its voice.json before use."""
        directory = Path(path)
        manifest = verify_package(directory)
        engine_type = ENGINES.get(manifest.engine)
        if engine_type is None:
            raise NotSupported(f"voice engine {manifest.engine!r} is not supported by this version of BytesTTS")
        config, defaults = _config(directory, manifest)
        frontend = _frontend(config)
        engine = engine_type(
            directory / "model.onnx",
            defaults,
            speakers=_speaker_ids(directory, manifest),
            device=device,
            threads=threads,  # unchanged: None keeps ONNX Runtime's defaults (P0.9)
        )
        card = read_json(directory / CARD, CARD) if CARD in manifest.files else None
        return cls(info=VoiceInfo.from_manifest(manifest), card=card, frontend=frontend, engine=engine)

    def stream(self, text: str, *, speaker: str | None = None, speed: float = 1.0) -> Iterator[Audio]:
        """One Audio per sentence, or per chunk of a long sentence. The arguments are checked at the call."""
        opts = SynthesisOptions(speed=speed)
        if speaker is not None and speaker not in self._engine.speakers:
            raise VoiceNotFound(f"speaker {speaker!r} is not in voice {self.info.id!r}")
        chunks = self._frontend.encode(text)
        return self._render(chunks, VoiceRef(self.info.id, speaker=speaker), opts)

    def _render(self, chunks: Sequence[Sequence[int]], voice: VoiceRef, opts: SynthesisOptions) -> Iterator[Audio]:
        for ids in chunks:
            yield Audio(self._engine.synthesize(ids, voice, opts), self._engine.sample_rate)

    def synthesize(self, text: str, *, speaker: str | None = None, speed: float = 1.0) -> Audio:
        """The whole text, its sentences joined by SENTENCE_PAUSE_S of silence."""
        return concat(list(self.stream(text, speaker=speaker, speed=speed)), pause_s=self.SENTENCE_PAUSE_S)
