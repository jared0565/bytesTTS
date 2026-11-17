# SPDX-License-Identifier: MPL-2.0
"""Exception hierarchy. Every error BytesTTS raises on purpose is a BytesTTSError."""


class BytesTTSError(Exception):
    """Base class for BytesTTS errors."""


class VoiceNotFound(BytesTTSError):
    """The requested voice or speaker does not exist."""


class IntegrityError(BytesTTSError):
    """A voice file failed verification (hash, size, layout or model format)."""


class FrontendUnavailable(BytesTTSError):
    """The text frontend cannot run, for example because espeak-ng is missing."""


class ConsentMissing(BytesTTSError):
    """A fine-tuned voice lacks its consent statement or voice card."""


class NotSupported(BytesTTSError):
    """The request needs a feature this voice, engine or device does not have."""


class NoSpeakableText(BytesTTSError, ValueError):
    """The input contains nothing that can be spoken."""
