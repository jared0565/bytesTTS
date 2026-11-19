# SPDX-License-Identifier: MPL-2.0
"""Package a converted voice as a BytesTTS voice package (host side; needs the [train] extra).

  uv run --extra train --extra torch-cpu python -P tools/convert/package_voice.py CONVERTED OUT \\
      --id en-ljspeech --version 1.0.0 --language en --licence ... --data-licence ... --attribution ... \\
      [--rename-speaker OLD=NEW ...] [--featured NAME ...]

CONVERTED is convert_coqui.py's output (generator.safetensors, model_config.json, voice_config.json and, for a
multi-speaker voice, speakers.json). OUT, which must be empty or absent, receives model.onnx, config.json,
speakers.json if any, and voice.json pinning them all. Weights are read from safetensors only: no pickle is
opened here. Runs single-threaded: all-core load resets the development PC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bytestts.catalog.manifest import FileEntry, VoiceManifest, sha256_file, verify_package
from bytestts.text.frontend import FrontendConfig

VOICE_CONFIG = "voice_config.json"
SPEAKERS = "speakers.json"


def _json_bytes(obj: Any) -> bytes:
    """The fixture tool's format: sorted keys, indent 2, ASCII, LF."""
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _speaker_ids(converted: Path, num_speakers: int) -> dict[str, int] | None:
    path = converted / SPEAKERS
    if not path.exists():
        if num_speakers:
            raise ValueError(f"the model has {num_speakers} speakers but {path} is missing")
        return None
    ids = json.loads(path.read_bytes())
    if not num_speakers:
        raise ValueError(f"{path} exists but the model has a single speaker")
    values = list(ids.values()) if isinstance(ids, dict) else []
    if (
        not values
        or not all(isinstance(k, str) and k for k in ids)
        or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < num_speakers for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{path} must map speaker names to distinct ids in 0..{num_speakers - 1}")
    return dict(ids)


class OptionError(ValueError):
    """A --rename-speaker or --featured value that cannot apply; the CLI refuses it with exit 2."""


def _renamed(ids: dict[str, int] | None, rename: dict[str, str]) -> dict[str, int]:
    """The speaker table with each OLD name replaced by NEW. The ids, and so the model rows, stay as they were."""
    if not ids:
        raise OptionError("--rename-speaker: the voice has a single speaker and no speaker table")
    for old, new in rename.items():
        if old not in ids:
            raise OptionError(f"--rename-speaker: no speaker named {old!r}")
        if not new or new != new.strip():
            raise OptionError(f"--rename-speaker: {new!r} is not a usable speaker name")
    renamed = {rename.get(name, name): i for name, i in ids.items()}
    if len(renamed) != len(ids):
        raise OptionError("--rename-speaker: a new name is already a speaker's name")
    return renamed


def _featured(ids: dict[str, int] | None, featured: list[str]) -> list[str]:
    """The featured speakers, in order. A package loaded from disk defaults to the first (api._speaker_ids)."""
    if featured and not ids:
        raise OptionError("--featured: the voice has a single speaker and no speaker table")
    for name in featured:
        if ids is not None and name not in ids:
            raise OptionError(f"--featured: no speaker named {name!r}")
        if featured.count(name) > 1:
            raise OptionError(f"--featured: {name!r} is listed twice")
    return list(featured)


def package(
    converted: Path,
    out: Path,
    *,
    voice_id: str,
    version: str,
    language: str,
    licence: str,
    data_licence: str,
    attribution: str,
    rename_speakers: dict[str, str] | None = None,
    featured: list[str] | None = None,
) -> VoiceManifest:
    from bytestts.train.vits.export import export_onnx
    from bytestts.train.vits.io import load_generator

    converted, out = Path(converted), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: a package is written into an empty directory only")
    voice_config = json.loads((converted / VOICE_CONFIG).read_bytes())
    config = {k: v for k, v in voice_config.items() if k != "model_args"}
    if config.get("language") != language:
        raise ValueError(f"language {language!r} differs from {VOICE_CONFIG}'s {config.get('language')!r}")

    model = load_generator(converted)
    vocabulary = len(FrontendConfig.from_voice_config(config).characters.vocab)
    if vocabulary != model.cfg.num_chars:  # else every token id would index the wrong embedding row
        raise ValueError(f"the text vocabulary has {vocabulary} symbols but the model embeds {model.cfg.num_chars}")
    ids = _speaker_ids(converted, model.cfg.num_speakers if model.cfg.use_speaker_embedding else 0)
    if rename_speakers:  # names only: speakers.json and the manifest change, the model and CONVERTED never do
        ids = _renamed(ids, rename_speakers)
    featured = _featured(ids, featured or [])  # checked against the new names, before anything is written

    out.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out / "model.onnx")
    (out / "config.json").write_bytes(_json_bytes(config))
    names = ["model.onnx", "config.json"]
    if ids:
        (out / SPEAKERS).write_bytes(_json_bytes(ids))
        names.append(SPEAKERS)
    manifest = VoiceManifest(
        id=voice_id,
        version=version,
        language=language,
        engine="vits-onnx",
        sample_rate=config["sample_rate"],
        licence=licence,
        data_licence=data_licence,
        attribution=attribution,
        origin="catalog",
        files={name: FileEntry(sha256_file(out / name), (out / name).stat().st_size) for name in names},
        speakers=sorted(ids, key=ids.__getitem__) if ids else [],
        featured=featured,  # the operator picks them by ear; the catalogue entry carries the same list
    )
    (out / "voice.json").write_bytes(_json_bytes(manifest.to_dict()))
    if verify_package(out) != manifest:
        raise RuntimeError(f"{out} does not verify as the package just written")
    return manifest


def _rename_pair(text: str) -> tuple[str, str]:
    old, sep, new = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"{text!r} is not OLD=NEW")
    return old, new


def main(argv: list[str] | None = None) -> int:
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:  # torch takes one set per process; a caller may have pinned it already
        torch.set_num_interop_threads(1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("converted", type=Path)
    ap.add_argument("out", type=Path)
    for flag in ("--id", "--version", "--language", "--licence", "--data-licence", "--attribution"):
        ap.add_argument(flag, required=True)
    ap.add_argument(
        "--rename-speaker",
        action="append",
        default=[],
        type=_rename_pair,
        metavar="OLD=NEW",
        help="rename one speaker (repeatable); its id, and so its row in the model, stays",
    )
    ap.add_argument(
        "--featured",
        action="append",
        default=[],
        metavar="NAME",
        help="a featured speaker, by its name after any rename (repeatable, in order); the first is the default",
    )
    args = ap.parse_args(argv)
    rename = dict(args.rename_speaker)
    if len(rename) != len(args.rename_speaker):
        ap.error("--rename-speaker: a speaker is named twice")
    try:
        m = package(
            args.converted,
            args.out,
            voice_id=args.id,
            version=args.version,
            language=args.language,
            licence=args.licence,
            data_licence=args.data_licence,
            attribution=args.attribution,
            rename_speakers=rename or None,
            featured=args.featured,
        )
    except OptionError as e:
        ap.error(str(e))
    print(f"{args.out}: {m.id} {m.version}, " + ", ".join(f"{n} {e.size:,} B" for n, e in m.files.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
