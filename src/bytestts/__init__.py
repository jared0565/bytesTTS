# SPDX-License-Identifier: MPL-2.0
"""BytesTTS: text-to-speech with VITS voices on ONNX Runtime."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from bytestts.errors import (
    BytesTTSError,
    ConsentMissing,
    FrontendUnavailable,
    IntegrityError,
    NoSpeakableText,
    NotSupported,
    VoiceNotFound,
)

__version__ = "0.1.0"

if TYPE_CHECKING:
    from bytestts.api import BytesTTS
    from bytestts.audio import Audio
    from bytestts.catalog.catalog import VoiceInfo, list_voices

# Imported on first use, so that `import bytestts.train...` (the converter, in an environment with torch but
# without the runtime's onnxruntime and requests) does not pull in the runtime.
_LAZY = {
    "BytesTTS": "bytestts.api",
    "Audio": "bytestts.audio",
    "VoiceInfo": "bytestts.catalog.catalog",
    "list_voices": "bytestts.catalog.catalog",
}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'bytestts' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "Audio",
    "BytesTTS",
    "BytesTTSError",
    "ConsentMissing",
    "FrontendUnavailable",
    "IntegrityError",
    "NoSpeakableText",
    "NotSupported",
    "VoiceInfo",
    "VoiceNotFound",
    "__version__",
    "list_voices",
]
