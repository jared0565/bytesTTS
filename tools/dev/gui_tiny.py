# SPDX-License-Identifier: MPL-2.0
"""The control panel with the tiny test voice, for trying the page by hand (control panel plan, Task 12).

  uv run python -P tools/dev/gui_tiny.py [--browser | --no-window] [--clip ok|down|stall|slow]

A temporary BYTESTTS_HOME holds the committed tiny fixture (tests/fixtures/tiny_vits) as the catalogue voice "tiny",
already downloaded and with no download URLs, beside the real catalogue. It is light: the tiny voice at 1 thread.

Sample clips (voice labels plan, Task 7): the shipped voice details are used, plus "tiny" as a voice with one
reader. Tiny's reader, and every voice's default speaker that has no pinned clip, get a short sine clip from a
server on 127.0.0.1, each voice at its own pitch, so the page's Sample and Preview can be tried with no upload and
no voice model. --clip picks how that server behaves: ok (the default) serves the clips; down points at a port
nothing listens on; stall sends the headers and never the body; slow delays tiny's clip by 3 s and serves the
others at once. Every step sits inside main().
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
    parser.add_argument(
        "--clip", choices=("ok", "down", "stall", "slow"), default="ok", help="how the local sample clips behave"
    )
    parser.set_defaults(mode="auto")
    args = parser.parse_args()
    tiny = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_vits"
    with tempfile.TemporaryDirectory(prefix="bytestts-gui-tiny-", ignore_cleanup_errors=True) as home:
        os.environ["BYTESTTS_HOME"] = home
        import dataclasses
        import hashlib
        import socket
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        import numpy as np

        from bytestts.audio import Audio
        from bytestts.catalog import catalog, details
        from bytestts.catalog.cache import voice_dir
        from bytestts.catalog.catalog import VoiceInfo
        from bytestts.catalog.manifest import FileEntry, VoiceManifest, safe_name
        from bytestts.gui.launch import LaunchOptions, run_panel

        fixture = VoiceManifest.from_json(tiny / "voice.json")
        files = {name: FileEntry(e.sha256, e.size, None) for name, e in fixture.files.items()}
        manifest = VoiceManifest(**{**fixture.__dict__, "files": files})
        folder = voice_dir(manifest)
        folder.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(tiny / name, folder / safe_name(name))
        voices = {**catalog.load_catalog(), manifest.id: manifest}
        catalog.load_catalog = lambda: dict(voices)  # type: ignore[assignment]

        rate = 22050
        t = np.arange(int(rate * 1.5), dtype=np.float32) / rate
        clips = {
            f"{voice_id}.wav": Audio(
                (0.3 * np.sin(2 * np.pi * (220 + 110 * i) * t)).astype(np.float32), rate
            ).to_wav_bytes()
            for i, voice_id in enumerate(voices)
        }

        class Clips(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                stall = self.path.startswith("/stall/")
                name = self.path.removeprefix("/stall/").lstrip("/")
                body = clips.get(name)
                if body is None:
                    self.send_error(404)
                    return
                if args.clip == "slow" and name == f"{manifest.id}.wav":
                    time.sleep(3)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.flush()
                    if stall:
                        time.sleep(3600)  # the headers and never the body: the page gives up after 30 s
                    self.wfile.write(body)
                except OSError:
                    pass  # the client gave up first

        server = ThreadingHTTPServer(("127.0.0.1", 0), Clips)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        if args.clip == "down":
            with socket.socket() as closed:  # bound and closed again, so nothing listens there
                closed.bind(("127.0.0.1", 0))
                base = f"http://127.0.0.1:{closed.getsockname()[1]}"
        elif args.clip == "stall":
            base += "/stall"

        def local(voice_id: str) -> FileEntry:
            data = clips[f"{voice_id}.wav"]
            return FileEntry(hashlib.sha256(data).hexdigest(), len(data), f"{base}/{voice_id}.wav")

        shipped = details.load_details()
        described = {**shipped.voices, manifest.id: details.VoiceDetails("English", "tiny", None, details.Person())}
        for voice_id, d in described.items():
            default = VoiceInfo.from_manifest(voices[voice_id]).default_speaker if voice_id in voices else None
            if default is None and d.reader is not None and d.reader.sample is None:
                d = dataclasses.replace(d, reader=dataclasses.replace(d.reader, sample=local(voice_id)))
            elif default is not None and d.speakers is not None:
                person = d.speakers.get(default, details.Person())
                if person.sample is None:
                    speakers = {**d.speakers, default: dataclasses.replace(person, sample=local(voice_id))}
                    d = dataclasses.replace(d, speakers=speakers)
            described[voice_id] = d
        found = details.Details(shipped.sentences, described)
        details.load_details = lambda: found  # type: ignore[assignment]
        print(f"gui_tiny: sample clips from {base} (--clip {args.clip})", flush=True)
        return run_panel(LaunchOptions(0, args.mode, 1))


if __name__ == "__main__":
    sys.exit(main())
