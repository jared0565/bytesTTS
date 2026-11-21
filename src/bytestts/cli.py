# SPDX-License-Identifier: MPL-2.0
"""The bytestts command line.

Exit codes: 0 OK; 1 any other failure (network, files, unexpected); 2 usage error or no speakable text;
3 voice not found, failed verification or missing consent; 4 frontend unavailable or not supported.
Errors print one line to stderr, never a traceback.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from bytestts import __version__
from bytestts.api import BytesTTS
from bytestts.catalog import details
from bytestts.catalog.catalog import list_voices
from bytestts.catalog.download import ensure_voice
from bytestts.catalog.manifest import verify_package
from bytestts.engines.base import MAX_SPEED, MIN_SPEED
from bytestts.errors import (
    ConsentMissing,
    FrontendUnavailable,
    IntegrityError,
    NoSpeakableText,
    NotSupported,
    VoiceNotFound,
)

EXIT_OK, EXIT_FAILURE, EXIT_USAGE, EXIT_VOICE, EXIT_UNSUPPORTED = 0, 1, 2, 3, 4


class _UsageError(Exception):
    pass


def _threads(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more (ONNX Runtime reads 0 as all cores)")
    return n


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be from 0 to 65535 (0 picks a free port)")
    return port


def _speed(value: str) -> float:
    try:
        speed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not MIN_SPEED <= speed <= MAX_SPEED:  # NaN fails this too
        raise argparse.ArgumentTypeError(f"must be between {MIN_SPEED} and {MAX_SPEED}")
    return speed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bytestts", description="Text to speech with BytesTTS voices.")
    parser.add_argument("--version", action="version", version=f"bytestts {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    say = commands.add_parser("say", help="speak text into a WAV file")
    say.add_argument("text", nargs="?", help="the text to speak, or - to read UTF-8 from standard input")
    say.add_argument("--file", type=Path, metavar="PATH", help="read the text from a UTF-8 file")
    voice = say.add_mutually_exclusive_group(required=True)
    voice.add_argument("--voice", metavar="ID", help="a catalogue voice, downloaded on first use")
    voice.add_argument("--voice-path", type=Path, metavar="DIR", help="a voice package directory")
    say.add_argument("--speaker", help="the speaker, for voices with several")
    say.add_argument("--speed", type=_speed, default=1.0, help=f"{MIN_SPEED} to {MAX_SPEED} (default 1.0)")
    say.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    say.add_argument("--threads", type=_threads, metavar="N", help="CPU threads for synthesis (default: all)")
    say.add_argument("-o", "--output", required=True, metavar="PATH", help="the WAV file to write, or - for stdout")
    say.set_defaults(run=_say)

    voices = commands.add_parser("voices", help="list the catalogue voices")
    voices.add_argument("--language", help="only voices for this language, such as en or de")
    voices.set_defaults(run=_voices)

    speakers = commands.add_parser("speakers", help="list a voice's speakers, the featured ones first")
    which = speakers.add_mutually_exclusive_group(required=True)
    which.add_argument("voice", nargs="?", metavar="VOICE", help="a catalogue voice (nothing is downloaded)")
    which.add_argument("--voice-path", type=Path, metavar="DIR", help="a voice package directory")
    speakers.set_defaults(run=_speakers)

    download = commands.add_parser("download", help="download catalogue voices ahead of use")
    target = download.add_mutually_exclusive_group(required=True)
    target.add_argument("id", nargs="?", help="the voice to download")
    target.add_argument("--all", action="store_true", help="every catalogue voice")
    download.set_defaults(run=_download)

    gui = commands.add_parser("gui", help="open the control panel (the app window on Windows, else the browser)")
    gui.add_argument("--port", type=_port, default=0, metavar="N", help="port on 127.0.0.1 (default: a free one)")
    where = gui.add_mutually_exclusive_group()
    where.add_argument("--browser", dest="mode", action="store_const", const="browser", help="open the browser")
    where.add_argument(
        "--no-window", dest="mode", action="store_const", const="none", help="print the address and serve only"
    )
    gui.add_argument(
        "--max-threads", type=_threads, metavar="N", help="thread cap (default: $BYTESTTS_GUI_MAX_THREADS, else all)"
    )
    gui.set_defaults(run=_gui, mode="auto")
    return parser


def _read_text(args: argparse.Namespace) -> str:
    if (args.text is None) == (args.file is None):
        raise _UsageError("give the text, - for standard input, or --file PATH (one of them)")
    if args.file is None and args.text != "-":
        return str(args.text)
    try:
        # bytes, never sys.stdin: its decoding follows the console code page
        data = sys.stdin.buffer.read() if args.file is None else Path(args.file).read_bytes()
    except OSError as e:
        raise _UsageError(f"cannot read {args.file}: {e.strerror or e}") from None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise _UsageError(f"{args.file or 'standard input'} is not UTF-8 text") from None


def _say(args: argparse.Namespace) -> int:
    text = _read_text(args)
    if args.voice_path is not None:
        tts = BytesTTS.load_path(args.voice_path, device=args.device, threads=args.threads)
    else:
        tts = BytesTTS.load(args.voice, device=args.device, threads=args.threads)
    wav = tts.synthesize(text, speaker=args.speaker, speed=args.speed).to_wav_bytes()
    if args.output == "-":
        sys.stdout.buffer.write(wav)
        sys.stdout.buffer.flush()
    else:
        Path(args.output).write_bytes(wav)
    return EXIT_OK


def _voices(args: argparse.Namespace) -> int:
    for v in list_voices(args.language):
        count = max(1, len(v.speakers))  # an empty speaker table is one speaker
        speakers = f"{count} speaker" if count == 1 else f"{count} speakers"
        print(f"{v.id}\t{v.language}\t{speakers}\t{v.licence}\t{v.size_bytes / 1e6:.1f} MB")
    return EXIT_OK


def _speakers(args: argparse.Namespace) -> int:
    """One speaker per line, in three tab-separated columns: the name; "featured" for the shortlist; and what the
    voice's published sources say about the speaker (`Female, 22 · English (England), Southern England`), empty for
    a package or a voice without details. The shortlist comes first, in its order, then the rest in the voice's
    order: the panel's order, so line 1 is the default speaker. A catalogue voice is read from the shipped catalogue
    only, and a package from its verified voice.json: nothing is downloaded and no model is loaded."""
    described: dict[str, str] = {}
    if args.voice_path is not None:
        manifest = verify_package(args.voice_path)
        names, featured = manifest.speakers, manifest.featured
    else:
        info = next((v for v in list_voices() if v.id == args.voice), None)
        if info is None:
            raise VoiceNotFound(f"unknown voice {args.voice}")
        names, featured = info.speakers, info.featured
        if (d := details.load_details().voice(info.id)) is not None:
            described = {name: p.describe() for name in names if (p := d.person(name)) is not None}
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    for name in [*featured, *(s for s in names if s not in featured)]:
        line = f"{name}\t{'featured' if name in featured else ''}\t{described.get(name, '')}"
        print(line.encode(encoding, errors="replace").decode(encoding))  # "?" for " · " on a narrow console
    return EXIT_OK


def _download(args: argparse.Namespace) -> int:
    ids = [v.id for v in list_voices()] if args.all else [args.id]
    for voice_id in ids:
        print(f"{voice_id}\t{ensure_voice(voice_id)}")
    return EXIT_OK


def _gui(args: argparse.Namespace) -> int:
    if any(importlib.util.find_spec(name) is None for name in ("fastapi", "uvicorn")):
        raise NotSupported('the control panel needs fastapi and uvicorn: pip install "bytestts[gui]"')
    from bytestts.gui import launch  # the [gui] extra: imported only for this command

    try:
        cap = launch.thread_cap(args.max_threads, os.environ, os.cpu_count() or 1)
    except ValueError as e:
        raise _UsageError(str(e)) from None
    return launch.run_panel(launch.LaunchOptions(args.port, args.mode, cap))


_EXIT_CODES: tuple[tuple[tuple[type[Exception], ...], int], ...] = (
    ((NoSpeakableText,), EXIT_USAGE),
    ((VoiceNotFound, IntegrityError, ConsentMissing), EXIT_VOICE),
    ((FrontendUnavailable, NotSupported), EXIT_UNSUPPORTED),
)


def _one_line(error: BaseException) -> str:
    return " ".join(line.strip() for line in str(error).splitlines() if line.strip()) or type(error).__name__


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # argparse: 0 after --help or --version, 2 on a usage error
        return e.code if isinstance(e.code, int) else EXIT_USAGE
    run: Callable[[argparse.Namespace], int] = args.run
    try:
        return run(args)
    except _UsageError as e:
        print(f"bytestts {args.command}: error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # the contract: one line on stderr, never a traceback
        code = next((code for kinds, code in _EXIT_CODES if isinstance(e, kinds)), EXIT_FAILURE)
        print(f"bytestts: error: {_one_line(e)}", file=sys.stderr)
        return code
