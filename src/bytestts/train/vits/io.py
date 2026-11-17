# SPDX-License-Identifier: MPL-2.0
# Derived from Coqui TTS v0.22.0 (the generator's checkpoint layout); the format here is safetensors only
"""Save and load a VitsGenerator as generator.safetensors + model_config.json. No pickle, strict keys."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from safetensors.torch import load_file, save_file

from bytestts.train.vits.config import VitsModelConfig
from bytestts.train.vits.model import VitsGenerator

WEIGHTS = "generator.safetensors"
CONFIG = "model_config.json"


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_generator(model: VitsGenerator, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    _atomic_write(directory / WEIGHTS, lambda p: save_file(state, str(p)))
    text = json.dumps(model.cfg.to_dict(), indent=2, sort_keys=True) + "\n"
    _atomic_write(directory / CONFIG, lambda p: p.write_text(text, "utf-8"))


def load_generator(directory: Path) -> VitsGenerator:
    cfg = VitsModelConfig.from_dict(json.loads((directory / CONFIG).read_text("utf-8")))
    model = VitsGenerator(cfg)
    model.load_state_dict(load_file(str(directory / WEIGHTS), device="cpu"), strict=True)
    return model.eval()
