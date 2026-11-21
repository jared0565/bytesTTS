# Troubleshooting

Find the message you see, or the CLI exit code, and follow the fix. Every error BytesTTS raises on purpose is a
`BytesTTSError` subclass with a one-line message; the CLI prints it after `error:` and exits with the code below.

| Exit code | Exception | Section |
|---|---|---|
| 2 | `NoSpeakableText`, usage errors | [Text and usage](#text-and-usage) |
| 3 | `VoiceNotFound`, `IntegrityError`, `ConsentMissing` | [Voices](#voices) |
| 4 | `FrontendUnavailable`, `NotSupported` | [espeak-ng](#espeak-ng), [GPU](#gpu), [Unsupported voices](#unsupported-voices), [Control panel](#control-panel) |
| 1 | anything else | [Other failures](#other-failures) |
| 130 | Ctrl+C | |

## Text and usage

**`the text has nothing this voice can speak`** (`NoSpeakableText`, exit 2)
The text is empty, only whitespace or punctuation, or made only of characters the voice has no symbol for. Check
what you passed; with `bytestts say -`, make sure something is actually piped in.

**`character '…' (U+…) is not in the voice's vocabulary; dropped`** (a logged warning)
The character was removed and the rest was spoken. It is logged once per character, and the text itself is never
logged. Emoji and symbols from other scripts are the usual cause. Replace them before synthesis if they matter.
With the Spanish and French voices, digits and `¿ ¡ « »` are dropped this way too: write numbers as words.

**`speed must be between 0.5 and 2.0, not …`** (`ValueError`; exit 2 from the CLI)
Use a speed from 0.5 (slower) to 2.0 (faster).

**`threads must be an int >= 1 or None, not …`** (`ValueError`)
`threads=0` is refused because ONNX Runtime would read it as "all cores". Use `None` for ONNX Runtime's default.

**Numbers or symbols are read oddly.** The English voice expands numbers, currency, ordinals, years and common
abbreviations; `:` and `;` become pauses, so `3:30` is read "three, thirty". Write the text the way it should be
said ("half past three") when that matters. The [User Guide](../user-guide.md#7-how-text-is-spoken) lists the rules.

**Non-ASCII text comes out wrong from the CLI.** `bytestts say` reads text as UTF-8 bytes. Save files as UTF-8 (a
BOM is fine). On Windows, `echo` in `cmd.exe` writes in the console code page, so prefer `--file` for non-ASCII
text.

## Voices

**`unknown voice …`** (`VoiceNotFound`, exit 3)
The id is not in the catalogue. Run `bytestts voices` to see what is available (today: `en-ljspeech`, `es-css10`
and `fr-css10`). For a voice
package directory, use `--voice-path` (or `BytesTTS.load_path`) instead of `--voice`.

**`… is not cached and offline mode is on`** (`VoiceNotFound`)
`BytesTTS.load(..., offline=True)` never downloads. Download the voice once with `bytestts download ID` while
online.

**`speaker '…' is not in this voice`** / **`… this voice has a single speaker`** (`VoiceNotFound`)
Check `tts.info.speakers`. A single-speaker voice takes no `speaker` argument.

**`… is missing`**, **`… size is … bytes, the manifest pins …`**, **`… sha256 does not match the manifest`**
(`IntegrityError`, exit 3)
A file in the voice package was deleted, truncated or changed after the package was made. Re-copy or re-download
the package. For a cached catalogue voice, delete its directory under the [cache](../user-guide.md#the-cache); it is
downloaded again on next use.

**`… is a symbolic link, which is refused`**, **`unsafe file name …`**, **`voice.json does not list …`**
(`IntegrityError`)
Voice packages must hold real files with plain names, all listed in `voice.json`. Copy the files rather than
linking them.

**`config.json: bad or missing text settings (…)`**, **`… synthesis settings`**, **`… must be a JSON object`**
(`IntegrityError`)
The package's `config.json` (or `card.json`) is damaged or was not made by BytesTTS's packager. Rebuild the
package.

**`… read … bytes, the manifest pins …`** (`IntegrityError`)
The file changed size while it was being read. Something else is writing to the voice directory; stop it and try
again.

**`the model stores tensors as external data …`**, **`the model is in ORT format …`**, **`malformed ONNX model …`**
(`IntegrityError`)
BytesTTS loads only self-contained ONNX models. It refuses models that point at files outside themselves, as a
safety rule (see [Security model](security-model.md)). Re-export the model as a single `.onnx` file.

**`this model takes a speaker id, but its voice lists no speakers`** (or the reverse) (`IntegrityError`)
The model and the package's `speakers.json` disagree. The package is broken; rebuild it.

**`a fine-tuned voice must list card.json …`**, **`card.json has no consent statement`** (`ConsentMissing`, exit 3)
A voice marked as fine-tuned must carry a voice card with the speaker's consent statement, pinned in `voice.json`.
BytesTTS will not load it without one.

**`… the download does not match its pinned size and sha256`**, **`… sent more than the pinned … bytes`**
(`IntegrityError`)
The server sent something other than the published file. Nothing was kept. Try again later. If it repeats, report it
as described in [SECURITY.md](../../SECURITY.md).

## espeak-ng

**`espeak-ng not found.`**, **`espeak-ng executable not found at …`**, **`no espeak-ng-data under …`**,
**`espeak-ng failed (exit code …)`**, **`espeak-ng could not run: …`** (`FrontendUnavailable`, exit 4)
See [Installing espeak-ng](espeak-ng.md#common-problems). On Windows, unpacking it into
`%LOCALAPPDATA%\bytestts-tools\espeak-ng` is enough; installed elsewhere, set `BYTESTTS_ESPEAK` (and, for a custom
Linux build, `BYTESTTS_ESPEAK_DATA`).

## GPU

**`CUDA execution provider not available`** (`NotSupported`, exit 4)
`device="cuda"` was asked for, but the installed ONNX Runtime has no CUDA support. Install the `gpu` extra and
reinstall `onnxruntime-gpu`, as in [GPU and performance](gpu-and-performance.md).

**`the CUDA execution provider is listed but the session runs on CPU …`** (`NotSupported` with `device="cuda"`, a
warning with `device="auto"`)
The GPU build is installed but cannot start, usually because the CUDA 13 runtime or cuDNN 9 is missing or not on
the library path. See [GPU and performance](gpu-and-performance.md#checking-the-gpu).

## Unsupported voices

**`voice engine '…' is not supported by this version of BytesTTS`**, **`text cleaner '…' is not supported`**,
**`unsupported characters_class …`**, **`phonemizer '…' is not supported; only espeak-ng is`** (`NotSupported`,
exit 4)
The voice needs something this version of BytesTTS lacks. Upgrade BytesTTS, or use a voice made for this version.

**`this engine cannot clone a voice from reference audio`** (`NotSupported`)
VITS voices speak only as their trained speakers. Voice cloning is not available.

## Control panel

**`bytestts gui: the app window needs pywebview …; opening the browser instead`**, **`bytestts gui: the app window
could not start: …`**, **`… did not load within 30 s`** (the panel opens in the browser instead)
The window needs pywebview and the Microsoft Edge WebView2 Runtime. Install the extra with
`pip install "bytestts[gui] @ git+https://github.com/jared0565/bytesTTS"`. If the message names WebView2, install
the WebView2 Runtime from Microsoft (Windows 11 includes it). The panel works the same in the browser meanwhile.

**`the control panel needs fastapi and uvicorn: …`** (`NotSupported`, exit 4)
The `gui` extra is not installed. Install it as above.

**No browser opens** (on Linux or WSL with no default browser; the terminal may show
`gio: http://127.0.0.1:…: Operation not supported`)
The panel is running, and its printed address has not been used: open it yourself in a browser, once. To have
`bytestts gui` open a browser itself, set the `BROWSER` environment variable to the browser's command first, for
example `BROWSER=firefox bytestts gui`.

**`cannot listen on 127.0.0.1:N: …`** (exit 1)
Another program, or another copy of the panel, is using that port. Leave out `--port` to let the system pick a free
one, or choose another number.

**`--max-threads must be from 1 to N …`**, **`BYTESTTS_GUI_MAX_THREADS must be a whole number from 1 to N …`**
(exit 2)
The thread cap must be between 1 and your number of CPU cores. Fix the option, or the variable in your environment.

**The GPU switch is greyed out.** ONNX Runtime does not offer CUDA. The reason is shown beside the switch: install
the `gpu` extra with the CUDA 13 runtime and cuDNN 9, as in [GPU and performance](gpu-and-performance.md).

**The Threads box stops at a low number.** It stops at the panel's thread cap, shown beside it as "(max N)". The cap
comes from `--max-threads` or `BYTESTTS_GUI_MAX_THREADS`; without either it is the number of CPU cores. See the
[User Guide](../user-guide.md#threads).

**`This page is not connected to the running panel: switch to the panel's own tab or window, or start it again with bytestts gui.`**
The page does not hold the running panel's secret: it was opened without the launch address (typed, or picked from
the browser's history), or the panel was restarted since (each launch makes a new secret). Switch to the tab or
window the panel is already open in. Do not open the printed address again: in the browser, the panel has already
used it, and a second use stops the panel. With `--no-window`, if you have not opened the printed address yet, open
it now. Otherwise start the panel again with `bytestts gui`.

**`wrong launch code: switch to the panel's own tab or window, or start it again with bytestts gui`**
The address is from an earlier launch, usually one that used the same `--port`, or it was changed. Switch to the tab
or window the panel is already open in, or start it again with `bytestts gui`. With `--no-window`, open the address
it printed, if you have not opened it yet.

**`this launch address was already used, so the panel has stopped: start it again with bytestts gui`**
The printed address works once, and it was opened a second time: in a second tab, or after the first tab was closed.
The panel stopped, the terminal printed `bytestts gui: the launch address was used a second time, so the panel
stopped; start it again with bytestts gui`, and `bytestts gui` exited with code 1. Run `bytestts gui` again: it
opens the panel itself (with `--no-window`, open its new address once). A reload of the first tab does not count as
a second use.

**`That file is not UTF-8 text`**, **`… is larger than 1 MB`**
Open text file reads UTF-8 text files of up to 1 MB. Save the file as UTF-8 in your editor, or split a longer text.

**`could not fetch the sample: …`** (after ▶ Sample or ▶ Preview)
A clip is fetched from huggingface.co the first time it is played, and the rest of the line says why that
failed: no network, a proxy or firewall in the way, Hugging Face unavailable, or `timed out` when nothing
arrived within 30 s. Try again later: once a clip has played it is cached and needs no network. A clip whose
bytes don't match its pinned SHA-256 is refused the same way and never played. On a downloaded voice ▶ Preview
speaks the sentence live instead and says `Couldn't fetch the sample, so speaking it live`.

**`Wait for the running task to finish, then press Preview again`**
Speaking a Preview live needs the CPU, and a Speak or another Preview is running. Press Preview again when it
has finished; at Speed 1.00, a speaker with a recorded clip plays it without waiting.

**`No speaker matches these filters: set one back to Any`**
No speaker of the voice has both the Gender and the Accent chosen. Set one of them back to *Any*.

**`the Save dialog is already open`** (in the window)
Finish or cancel the Save dialog that is open; it may be behind the window.

## Other failures

**Exit 1 with a network message.** A download failed: no connection, a proxy, or a TLS problem.

**`refusing to download …: https only …`** Downloads use https, and every redirect is checked before it is
followed. A redirect from https back to http, to `localhost` or to a private IP address is refused. Plain http is
allowed only to `127.0.0.1` or `localhost`, for local mirrors. See [Security model](security-model.md#downloads).

**It is slow, or the PC becomes unresponsive.** See [GPU and performance](gpu-and-performance.md).

**Still stuck?** [Open an issue](https://github.com/jared0565/bytesTTS/issues/new/choose) with the full `error:`
line, `bytestts --version`, your Python version, your operating system and, for phoneme voices,
`espeak-ng --version`. Leave out any private text.
