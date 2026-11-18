# SPDX-License-Identifier: MPL-2.0
"""The control panel with the tiny test voice, for trying the page by hand (control panel plan, Task 12).

  uv run python -P tools/dev/gui_tiny.py [--browser | --no-window]

A temporary BYTESTTS_HOME holds the committed tiny fixture (tests/fixtures/tiny_vits) as the catalogue voice "tiny",
already downloaded and with no download URLs, beside the real catalogue. It is light: the tiny voice at 1 thread.
Every step sits inside main().
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="The control panel with the tiny test voice.")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--browser", dest="mode", action="store_const", const="browser", help="open the browser")
    where.add_argument("--no-window", dest="mode", action="store_const", const="none", help="print the address only")
    parser.set_defaults(mode="auto")
    args = parser.parse_args()
    tiny = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_vits"
    with tempfile.TemporaryDirectory(prefix="bytestts-gui-tiny-", ignore_cleanup_errors=True) as home:
        os.environ["BYTESTTS_HOME"] = home
        from bytestts.catalog import catalog
        from bytestts.catalog.cache import voice_dir
        from bytestts.catalog.manifest import FileEntry, VoiceManifest, safe_name
        from bytestts.gui.launch import LaunchOptions, run_panel

        fixture = VoiceManifest.from_json(tiny / "voice.json")
        files = {name: FileEntry(e.sha256, e.size, None) for name, e in fixture.files.items()}
        manifest = VoiceManifest(**{**fixture.__dict__, "files": files})
        folder = voice_dir(manifest)
        folder.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(tiny / name, folder / safe_name(name))
        real = catalog.load_catalog
        catalog.load_catalog = lambda: {**real(), manifest.id: manifest}  # type: ignore[assignment]
        return run_panel(LaunchOptions(0, args.mode, 1))


if __name__ == "__main__":
    sys.exit(main())
