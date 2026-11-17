# SPDX-License-Identifier: MPL-2.0
"""BytesTTS: text-to-speech with VITS voices on ONNX Runtime."""

__version__ = "0.1.0"

from bytestts.api import BytesTTS
from bytestts.audio import Audio
from bytestts.catalog.catalog import VoiceInfo, list_voices
from bytestts.errors import (
    BytesTTSError,
    ConsentMissing,
    FrontendUnavailable,
    IntegrityError,
    NoSpeakableText,
    NotSupported,
    VoiceNotFound,
)

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
