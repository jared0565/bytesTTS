# SPDX-License-Identifier: MPL-2.0
"""Golden traces from TTS-dev 0.22.0 (torch 2.5.1, in the sandbox).

The module-level names are en-ljspeech's (P0.7, espeak-ng 1.52.0). The CSS10 Spanish and French traces (2026-11-18)
load by voice id.
"""

import json
from pathlib import Path
from typing import Any

GOLDEN_ROOT = Path(__file__).resolve().parent / "fixtures" / "golden"


def golden_dir(voice_id: str) -> Path:
    return GOLDEN_ROOT / voice_id


def load_trace(voice_id: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((golden_dir(voice_id) / "trace.json").read_text("utf-8"))
    return data


GOLDEN_DIR = golden_dir("en-ljspeech")
GOLDEN = GOLDEN_DIR / "trace.json"
trace: dict[str, Any] = load_trace("en-ljspeech")
VOICE_CFG: dict[str, Any] = trace["voice_config"]
