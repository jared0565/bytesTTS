# SPDX-License-Identifier: MPL-2.0
"""Where downloaded voices live."""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

from bytestts.catalog.manifest import VoiceManifest, safe_name

HOME_ENV = "BYTESTTS_HOME"


def cache_root() -> Path:
    """BYTESTTS_HOME, else the user cache directory (on Windows %LOCALAPPDATA%\\bytestts\\Cache)."""
    home = os.environ.get(HOME_ENV)
    if home:
        return Path(home).expanduser()
    return Path(platformdirs.user_cache_dir("bytestts", appauthor=False))


def voice_dir(m: VoiceManifest) -> Path:
    """Keyed by the manifest digest, so an updated voice never mixes with a stale one."""
    return cache_root() / "voices" / safe_name(m.id) / m.manifest_digest()[:16]
