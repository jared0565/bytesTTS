# SPDX-License-Identifier: MPL-2.0
"""Regenerate the committed test fixtures (Task 10).

  uv run --extra train --extra torch-cpu python -P tools/dev/make_tiny_fixture.py tests/fixtures/tiny_vits
  uv run --extra train --extra torch-cpu python -P tools/dev/make_tiny_fixture.py --hostile tests/fixtures/hostile

The tiny voice is a seeded (1234) tiny VITS with the deterministic duration predictor. The stochastic one alone
is 5.3 MB per weight copy and the fixture must stay under 2 MB. Its weights are perturbed by a seeded
N(0, 0.05) so the zero-initialized flows are not identities. It contains:
  voice.json   the manifest (Task 12), pinning model.onnx and config.json by sha256 and size
  model.onnx   export_onnx's graph
  config.json  the characters frontend (tests/helpers.py CHAR_CFG) and the synthesis defaults
  model_config.json   the generator architecture (VitsModelConfig.to_dict)
  golden.npz   3 token sequences with the PyTorch port's durations and waveform at scales [0, 1, 0]
`--with-generator` also writes generator.safetensors (1 MB; nothing reads it, so it is not committed).

Every file is written as bytes with LF line ends; .gitattributes keeps git from converting them.
Runs single-threaded: all-core load resets the development PC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SEED = 1234
PERTURB_SEED = 3
SAMPLE_RATE = 22050
SENTENCES = ("hello there.", "the quick brown fox jumps over the lazy dog!", "a, b?")
MARGIN = 1e-3  # unrounded durations must sit this far from an integer, or runtimes may round differently
TINY_ARGS: dict[str, Any] = {  # == tests/helpers.py TINY_ARGS (tests/test_tiny_fixture.py checks)
    "out_channels": 17,
    "hidden_channels": 16,
    "hidden_channels_ffn_text_encoder": 32,
    "num_heads_text_encoder": 2,
    "num_layers_text_encoder": 1,
    "num_layers_posterior_encoder": 1,
    "num_layers_flow": 1,
    "resblock_kernel_sizes_decoder": [3],
    "resblock_dilation_sizes_decoder": [[1, 3, 5]],
    "upsample_rates_decoder": [4, 4],
    "upsample_initial_channel_decoder": 32,
    "upsample_kernel_sizes_decoder": [8, 8],
}
VOICE_CONFIG: dict[str, Any] = {  # P0.7 voice_config keys; the frontend must equal tests/helpers.py CHAR_CFG
    "language": "en",
    "use_phonemes": False,
    "phoneme_language": None,
    "phonemizer": None,
    "text_cleaner": "basic_cleaners",
    "characters": {
        "characters_class": "TTS.tts.utils.text.characters.Graphemes",
        "characters": "abcdefghijklmnopqrstuvwxyz",
        "punctuations": "!'(),-.:;? ",
        "pad": "<PAD>",
        "eos": None,
        "bos": None,
        "blank": "<BLNK>",
        "is_unique": True,
        "is_sorted": True,
    },
    "add_blank": True,
    "enable_eos_bos_chars": False,
    "sample_rate": SAMPLE_RATE,
    "inference_noise_scale": 0.667,
    "inference_noise_scale_dp": 1.0,
    "length_scale": 1.0,
}


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _entry(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def make_tiny(out: Path, *, with_generator: bool) -> None:
    from bytestts.text.frontend import FrontendConfig, TextFrontend
    from bytestts.train.vits.config import VitsModelConfig
    from bytestts.train.vits.export import export_onnx
    from bytestts.train.vits.io import save_generator
    from bytestts.train.vits.model import VitsGenerator

    frontend_cfg = FrontendConfig.from_voice_config(VOICE_CONFIG)
    frontend = TextFrontend(frontend_cfg)
    num_chars = len(frontend_cfg.characters.vocab)

    torch.manual_seed(SEED)
    model = VitsGenerator(VitsModelConfig.from_coqui({**TINY_ARGS, "use_sdp": False}, num_chars=num_chars))
    g = torch.Generator().manual_seed(PERTURB_SEED)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    model.eval()

    golden: dict[str, np.ndarray] = {"count": np.array(len(SENTENCES))}
    for i, sentence in enumerate(SENTENCES):
        (ids,) = frontend.encode(sentence)
        x, lengths = torch.tensor([ids]), torch.tensor([len(ids)])
        with torch.no_grad():
            h, _, _, mask = model.text_encoder(x, lengths)
            w = (torch.exp(model.duration_predictor(h, mask)) * mask)[mask.bool()]
        margin = float(torch.min(torch.abs(w - torch.round(w))))
        if margin < MARGIN:
            sys.exit(f"sentence {sentence!r}: a duration is {margin:.1e} from an integer; pick another sentence")
        wav, dur = model.infer(x, lengths, noise_scale=0.0, length_scale=1.0, noise_scale_dp=0.0)
        golden[f"tokens_{i}"] = np.array(ids, np.int64)
        golden[f"durations_{i}"] = dur[0].numpy()
        golden[f"wav_{i}"] = wav[0, 0].numpy()
        print(f"golden {i}: {len(ids)} tokens, {int(dur.sum())} frames, {wav.shape[-1]} samples, margin {margin:.1e}")

    out.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out / "model.onnx")
    (out / "config.json").write_bytes(_json_bytes(VOICE_CONFIG))
    (out / "model_config.json").write_bytes(_json_bytes(model.cfg.to_dict()))
    np.savez(out / "golden.npz", **golden)
    if with_generator:
        save_generator(model, out)
    voice = {
        "id": "tiny",
        "version": "0.1.0",
        "language": "en",
        "engine": "vits-onnx",
        "sample_rate": SAMPLE_RATE,
        "licence": "MPL-2.0",
        "data_licence": "none",
        "attribution": "BytesTTS test fixture: a seeded random tiny VITS. No training data, no real voice.",
        "origin": "catalog",
        "files": {name: _entry(out / name) for name in ("model.onnx", "config.json")},
        "speakers": [],
        "featured": [],
    }
    (out / "voice.json").write_bytes(_json_bytes(voice))
    sizes = {f.name: f.stat().st_size for f in sorted(out.iterdir())}
    print(f"{out}: {sum(sizes.values()):,} bytes", sizes)


def make_hostile(out: Path) -> None:
    """external_data.onnx: one Add node whose initializer points at ../secret.bin as external data."""
    from onnx import TensorProto, helper

    w = TensorProto(name="w", data_type=TensorProto.FLOAT, dims=[1], data_location=TensorProto.EXTERNAL)
    for key, value in (("location", "../secret.bin"), ("offset", "0"), ("length", "4")):
        entry = w.external_data.add()
        entry.key, entry.value = key, value
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "w"], ["y"])],
        "hostile",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    out.mkdir(parents=True, exist_ok=True)
    (out / "external_data.onnx").write_bytes(model.SerializeToString())  # no checker: it would resolve the path
    print(f"{out / 'external_data.onnx'}: {(out / 'external_data.onnx').stat().st_size} bytes")


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--hostile", action="store_true", help="write the hostile external-data model instead")
    ap.add_argument("--with-generator", action="store_true", help="also write generator.safetensors")
    args = ap.parse_args(argv)
    if args.hostile:
        make_hostile(args.out)
    else:
        make_tiny(args.out, with_generator=args.with_generator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
