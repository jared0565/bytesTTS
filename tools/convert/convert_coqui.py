# SPDX-License-Identifier: MPL-2.0
"""Convert a Coqui VITS release archive into BytesTTS's training layout. SANDBOX ONLY: it opens a pickle.

Run it under SBX (the bubblewrap jail: no network, no Windows files, read-only inbox), never on the host:

  python -P convert_coqui.py ARCHIVE OUT [--sources sources.json] [--language LANG] [--allow-full-load]

1. ARCHIVE's sha256 and size must match its entry in sources.json (found by file name).
2. Only the checkpoint (model_file.pth[.tar]), config.json and, for a multi-speaker voice, speaker_ids.json are
   read from the zip; nothing is extracted to disk except the checkpoint, into a private temporary directory.
3. picklescan must report no issue and no global outside EXPECTED_GLOBALS (P0.5 found only these three).
4. torch.load(weights_only=True). P0.5: every v1 candidate loads this way, so the allow-list is empty. If it
   fails, exit 3, unless --allow-full-load (a full unpickle, inside the jail only), which the report records.
5. convert_state (no pickle; tested on the host): the state dict is split (generator / disc / dropped), legacy
   weight-norm keys renamed, and loaded STRICTLY into the port; the token vocabulary must equal the embedding's
   row count.

Writes OUT/generator.safetensors and model_config.json (save_generator), discriminator.safetensors (if the
checkpoint has one), voice_config.json, speakers.json (multi-speaker) and conversion_report.json.
Exit codes: 0 converted; 2 refused to run (host, usage, OUT not empty); 3 weights_only load failed;
4 integrity (hash, size, scan, members, keys, shapes, vocabulary); 5 an unsupported voice kind (d-vector,
language embedding). Nothing is written to OUT unless the conversion succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPECTED_GLOBALS = frozenset({"collections.OrderedDict", "torch.FloatStorage", "torch._utils._rebuild_tensor_v2"})
EXIT_REFUSED, EXIT_WEIGHTS_ONLY, EXIT_INTEGRITY, EXIT_UNSUPPORTED = 2, 3, 4, 5


class Refused(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _source(archive: Path, sources: Path) -> tuple[str, dict[str, Any]]:
    entries = json.loads(sources.read_bytes())
    matches = [(key, e) for key, e in entries.items() if e["url"].rsplit("/", 1)[1] == archive.name]
    if len(matches) != 1:
        raise Refused(EXIT_INTEGRITY, f"{archive.name} is not listed (once) in {sources}")
    key, entry = matches[0]
    if archive.stat().st_size != entry["size"] or _sha256(archive) != entry["sha256"]:
        raise Refused(EXIT_INTEGRITY, f"{archive.name}: size or sha256 differs from {sources}")
    return key, entry


def _members(z: zipfile.ZipFile) -> dict[str, str]:
    """Real members only: no directories, no macOS `__MACOSX/` or `._name` resource forks."""
    found: dict[str, list[str]] = {}
    for info in z.infolist():
        base = info.filename.rsplit("/", 1)[-1]
        if info.is_dir() or "__MACOSX/" in info.filename or base.startswith("._"):
            continue
        kind = "checkpoint" if base.endswith((".pth", ".pth.tar")) else base
        found.setdefault(kind, []).append(info.filename)
    out = {}
    for kind in ("checkpoint", "config.json", "speaker_ids.json"):
        names = found.get(kind, [])
        if len(names) > 1:
            raise Refused(EXIT_INTEGRITY, f"the archive has {len(names)} {kind} members: {names}")
        if names:
            out[kind] = names[0]
    if "checkpoint" not in out or "config.json" not in out:
        raise Refused(EXIT_INTEGRITY, "the archive lacks a checkpoint or config.json")
    return out


def _scan(checkpoint: Path) -> dict[str, Any]:
    from picklescan.scanner import scan_file_path

    scan = scan_file_path(str(checkpoint))
    names = sorted({f"{g.module}.{g.name}" for g in scan.globals})
    report = {"issues_count": scan.issues_count, "scan_err": scan.scan_err, "globals": names}
    unexpected = sorted(set(names) - EXPECTED_GLOBALS)
    if scan.issues_count or scan.scan_err or unexpected:
        raise Refused(EXIT_INTEGRITY, f"picklescan: {report}; unexpected globals {unexpected}")
    return report


def _require_sandbox() -> None:
    if os.name == "nt":
        raise Refused(EXIT_REFUSED, "convert_coqui.py opens pickles: run it in the sandbox (SBX), never on the host")


def _load(checkpoint: Path, allow_full_load: bool) -> tuple[Any, str]:
    _require_sandbox()  # here, not only in main(): no caller may unpickle on the host
    import torch

    try:
        return torch.load(str(checkpoint), map_location="cpu", weights_only=True), "weights_only"
    except Exception as e:  # any failure here means the safe loader refused
        if not allow_full_load:
            raise Refused(EXIT_WEIGHTS_ONLY, f"weights_only load failed ({type(e).__name__}: {e})") from None
    return torch.load(str(checkpoint), map_location="cpu", weights_only=False), "full (--allow-full-load)"


def _coqui_keys() -> Any:
    spec = importlib.util.spec_from_file_location("coqui_keys", HERE / "coqui_keys.py")
    if spec is None or spec.loader is None:
        raise Refused(EXIT_REFUSED, f"cannot load {HERE / 'coqui_keys.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_state(
    state: Mapping[str, Any], coqui_cfg: Mapping[str, Any], *, language: str, speaker_ids: Any, out: Path
) -> dict[str, Any]:
    """Everything after the unpickle: split, rename, check, strict-load into the port, write OUT.

    Works on an in-memory state dict and opens no file other than OUT's, so it is safe (and tested) on the host.
    Every refusal comes before OUT is created.
    """
    import safetensors.torch

    from bytestts.errors import NotSupported
    from bytestts.text.tokenizer import CharacterSet
    from bytestts.train.vits.config import VitsModelConfig
    from bytestts.train.vits.io import save_generator
    from bytestts.train.vits.model import VitsGenerator

    coqui_keys = _coqui_keys()
    generator, discriminator = coqui_keys.split_state_dict(state)
    dropped = sorted(k for k in state if not k.startswith(coqui_keys.DISC) and k not in generator)
    try:
        generator = coqui_keys.rename_legacy_weight_norm(generator)
        discriminator = coqui_keys.rename_legacy_weight_norm(discriminator)
    except ValueError as e:
        raise Refused(EXIT_INTEGRITY, f"weight-norm keys: {e}") from None

    voice_config = coqui_keys.extract_voice_config(coqui_cfg, language)
    embedding = generator.get("text_encoder.emb.weight")
    if embedding is None:
        raise Refused(EXIT_INTEGRITY, "the state dict does not fit the port: no text_encoder.emb.weight")
    num_chars = int(embedding.shape[0])
    vocabulary = len(CharacterSet.from_config(voice_config["characters"]).vocab)
    if vocabulary != num_chars:
        raise Refused(EXIT_INTEGRITY, f"the vocabulary has {vocabulary} symbols but the embedding has {num_chars}")
    try:
        cfg = VitsModelConfig.from_coqui(voice_config["model_args"], num_chars, sample_rate=voice_config["sample_rate"])
    except NotSupported as e:
        raise Refused(EXIT_UNSUPPORTED, str(e)) from None
    if cfg.use_speaker_embedding and not speaker_ids:
        raise Refused(EXIT_INTEGRITY, "a multi-speaker checkpoint without speaker_ids.json")
    model = VitsGenerator(cfg)
    try:
        model.load_state_dict(generator, strict=True)  # missing, unexpected and mis-shaped keys all raise
    except RuntimeError as e:
        raise Refused(EXIT_INTEGRITY, f"the state dict does not fit the port: {str(e)[:600]}") from None

    out.mkdir(parents=True, exist_ok=True)
    save_generator(model.eval(), out)
    if discriminator:
        tensors = {k: v.contiguous() for k, v in discriminator.items()}
        safetensors.torch.save_file(tensors, str(out / "discriminator.safetensors"))
    (out / "voice_config.json").write_bytes(_json_bytes(voice_config))
    if cfg.use_speaker_embedding:
        (out / "speakers.json").write_bytes(_json_bytes(speaker_ids))
    report = {
        "language": language,
        "dropped_keys": dropped,
        "num_chars": num_chars,
        "generator_keys": len(generator),
        "discriminator_keys": len(discriminator),
        "num_speakers": len(speaker_ids or {}) if cfg.use_speaker_embedding else 0,
    }
    (out / "conversion_report.json").write_bytes(_json_bytes(report))
    return report


def convert(archive: Path, out: Path, *, sources: Path, language: str | None, allow_full_load: bool) -> dict[str, Any]:
    """Pin, read, scan and load ARCHIVE (sandbox only: it unpickles), then convert_state."""
    import torch

    if out.exists() and any(out.iterdir()):
        raise Refused(EXIT_REFUSED, f"{out} is not empty")
    key, entry = _source(archive, sources)
    language = language or key.split("/", 1)[0]
    provenance: dict[str, Any] = {"archive": archive.name, "source": key, "sha256": entry["sha256"]}
    provenance["torch_version"] = torch.__version__

    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        members = _members(z)
        provenance["members"] = members
        coqui_cfg = json.loads(z.read(members["config.json"]))
        speaker_ids = json.loads(z.read(members["speaker_ids.json"])) if "speaker_ids.json" in members else None
        checkpoint = Path(tmp) / "checkpoint.pth"
        checkpoint.write_bytes(z.read(members["checkpoint"]))
        provenance["picklescan"] = _scan(checkpoint)
        obj, provenance["load_mode"] = _load(checkpoint, allow_full_load)

    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(state, Mapping):
        raise Refused(EXIT_INTEGRITY, f"the checkpoint holds a {type(state).__name__}, not a state dict")
    report = convert_state(state, coqui_cfg, language=language, speaker_ids=speaker_ids, out=out)
    report = {**provenance, **report}
    (out / "conversion_report.json").write_bytes(_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--sources", type=Path, default=HERE / "sources.json")
    ap.add_argument("--language", help="the voice's language (default: the first part of its sources key)")
    ap.add_argument("--allow-full-load", action="store_true", help="unpickle fully if weights_only fails")
    args = ap.parse_args(argv)
    try:
        _require_sandbox()
    except Refused as e:
        print(f"convert_coqui: {e}", file=sys.stderr)
        return e.code
    import torch

    torch.set_num_threads(1)
    try:
        report = convert(
            args.archive, args.out, sources=args.sources, language=args.language, allow_full_load=args.allow_full_load
        )
    except Refused as e:
        print(f"convert_coqui: {e}", file=sys.stderr)
        return e.code
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
