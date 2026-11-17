# SPDX-License-Identifier: MPL-2.0
"""The en-ljspeech golden trace from P0.7 (TTS-dev 0.22.0, torch 2.5.1, espeak-ng 1.52.0)."""

import json
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden" / "en-ljspeech"
GOLDEN = GOLDEN_DIR / "trace.json"
trace: dict[str, Any] = json.loads(GOLDEN.read_text("utf-8"))
VOICE_CFG: dict[str, Any] = trace["voice_config"]
