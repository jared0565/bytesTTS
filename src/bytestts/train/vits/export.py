# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0, TTS/tts/models/vits.py (Vits.export_onnx, lines 1806-1884)
"""Export a VitsGenerator to ONNX for the runtime engine (Task 11).

Graph I/O: `input` int64 [B, T], `input_lengths` int64 [B], `scales` float32 [3] (noise, length, noise_dp),
`sid` int64 [B] only for multi-speaker voices; `output` float32 [B, 1, S], `durations` int64 [B, T].

Exporter: the legacy TorchScript tracer (`dynamo=False`), the path TTS-dev uses (phase0-findings §2).
Unlike TTS-dev's export, all three scales are live inputs. TTS-dev's onnx_inference set `noise_scale` and
`noise_scale_dp` while `inference` reads `inference_noise_scale` and `inference_noise_scale_dp`
(vits.py:1829 vs 1133/1155), so its graphs carried the export-time noise as constants (P0.7 saw the noise
scales have no effect).
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from bytestts.train.vits.model import VitsGenerator

OPSET = 17
_DUMMY_TOKENS = 50


class _OnnxForward(nn.Module):
    def __init__(self, model: VitsGenerator) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input: Tensor, input_lengths: Tensor, scales: Tensor, sid: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        return self.model._infer(input, input_lengths, scales[0], scales[1], scales[2], sid)


def _plain_weights(model: VitsGenerator) -> VitsGenerator:
    """A copy with weight norm folded into plain weights (the same values; a smaller, faster graph).

    Built fresh and loaded from the state dict: deepcopy of a parametrized module loses its `weight`."""
    m = VitsGenerator(model.cfg)
    m.load_state_dict(model.state_dict(), strict=True)
    m.eval()
    for module in m.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return m


def export_onnx(model: VitsGenerator, path: Path) -> None:
    m = _plain_weights(model)
    multi_speaker = hasattr(m, "emb_g")
    g = torch.Generator().manual_seed(0)
    args: tuple[Tensor, ...] = (
        torch.randint(1, m.cfg.num_chars, (1, _DUMMY_TOKENS), generator=g),
        torch.tensor([_DUMMY_TOKENS], dtype=torch.long),
        torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32),
    )
    input_names = ["input", "input_lengths", "scales"]
    dynamic_axes = {
        "input": {0: "batch", 1: "phonemes"},
        "input_lengths": {0: "batch"},
        "output": {0: "batch", 2: "samples"},
        "durations": {0: "batch", 1: "phonemes"},
    }
    if multi_speaker:
        args += (torch.tensor([0], dtype=torch.long),)
        input_names.append("sid")
        dynamic_axes["sid"] = {0: "batch"}

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        with torch.no_grad(), warnings.catch_warnings():
            # the spline's domain checks are Python bools over tensors: evaluated once, as in TTS-dev's export
            warnings.simplefilter("ignore", getattr(torch.jit, "TracerWarning"))  # noqa: B009 -- not re-exported
            # legacy export is the recorded choice (phase0-findings P0.2; dynamo is the documented fallback):
            # its deprecation notices, attributed to this call or raised inside torch.onnx, are known
            warnings.filterwarnings("ignore", "You are using the legacy TorchScript", DeprecationWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.onnx")
            # torch.flip exports as a step -1 Slice, which the exporter declines to fold; harmless
            warnings.filterwarnings("ignore", message="Constant folding - Only steps=1", category=UserWarning)
            torch.onnx.export(
                _OnnxForward(m),
                args,
                tmp,
                opset_version=OPSET,
                dynamo=False,
                input_names=input_names,
                output_names=["output", "durations"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
        onnx.checker.check_model(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
