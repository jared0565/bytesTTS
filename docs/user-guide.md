# BytesTTS User Guide

This guide covers installing BytesTTS, speaking text from Python and from the command line, working with voices,
and configuring the library. For problems, see the [Knowledge Base](knowledge-base/README.md); for how it works
inside, see [Design](design.md).

- [1. Requirements](#1-requirements)
- [2. Installation](#2-installation)
- [3. Voices](#3-voices)
- [4. Python API](#4-python-api)
- [5. Command line](#5-command-line)
- [6. How text is spoken](#6-how-text-is-spoken)
- [7. Speed, speakers, devices and threads](#7-speed-speakers-devices-and-threads)
- [8. Voice packages and the cache](#8-voice-packages-and-the-cache)
- [9. Configuration reference](#9-configuration-reference)
- [10. Logging](#10-logging)

## 1. Requirements

- Windows or Linux, 64-bit.
- Python 3.11, 3.12, 3.13 or 3.14.
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) 1.52.0 for voices that speak from phonemes (`en-ljspeech`
  does). Voices that use plain characters do not need it.
- Optional: an NVIDIA GPU with the CUDA 13 runtime and cuDNN 9.

## 2. Installation

BytesTTS is not on PyPI yet. Install it from GitHub:

```
pip install "bytestts @ git+https://github.com/jared0565/bytesTTS"
```

or from a clone:

```
git clone https://github.com/jared0565/bytesTTS
cd bytesTTS
pip install .
```

This installs the `bytestts` package and the `bytestts` command.

### espeak-ng

**Windows** (no administrator rights needed):

1. Download `espeak-ng.msi` from the [1.52.0 release](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0).
2. Unpack it without installing it system-wide:

   ```
   msiexec /a espeak-ng.msi /qn TARGETDIR=%LOCALAPPDATA%\bytestts-tools\espeak-ng
   ```

3. Point BytesTTS at it (for the current window; use *System Properties > Environment Variables* to make it
   permanent):

   ```
   set BYTESTTS_ESPEAK=%LOCALAPPDATA%\bytestts-tools\espeak-ng\eSpeak NG\espeak-ng.exe
   ```

**Linux:** install your distribution's `espeak-ng` package (`sudo apt install espeak-ng` on Debian and Ubuntu), or
build the 1.52.0 tag from source for exactly the tested version.

How BytesTTS finds espeak-ng, and what to do when it doesn't, is in
[Installing espeak-ng](knowledge-base/espeak-ng.md).

### GPU

```
pip install "bytestts[gpu] @ git+https://github.com/jared0565/bytesTTS"
pip install --force-reinstall --no-deps onnxruntime-gpu
```

`onnxruntime` and `onnxruntime-gpu` install the same files, and whichever was installed last wins; the second
command makes sure the GPU build is the one on disk. NVIDIA's CUDA 13 runtime and cuDNN 9 are installed separately,
from NVIDIA. See [GPU and performance](knowledge-base/gpu-and-performance.md).

### Check the installation

```
bytestts --version
python -c "import bytestts; print(bytestts.__version__)"
```

## 3. Voices

A **voice** is a trained model plus the settings that turn text into its input. BytesTTS loads voices from **voice
packages**: a directory holding `voice.json` (the manifest), `model.onnx` and `config.json`, and `speakers.json` for
voices with several speakers. Every file is checked against the SHA-256 and size in `voice.json` before it is used.

There are two ways to get a voice:

- **The catalogue**: `BytesTTS.load("en-ljspeech")` or `bytestts say --voice en-ljspeech` downloads the voice on
  first use into the cache, checks every file's size and SHA-256 against the pins shipped inside BytesTTS, and
  loads it from the cache after that. `bytestts voices` lists the catalogue; `bytestts download ID` fetches a voice
  ahead of time.
- **A voice package directory** you already have: `BytesTTS.load_path(directory)` or
  `bytestts say --voice-path directory`.

Available voices:

| Voice | Language | Speakers | Size | Model licence | Data licence | Files |
|---|---|---|---|---|---|---|
| `en-ljspeech` | English (US) | 1 | 114.3 MB | Apache-2.0 | Public domain | [Hugging Face](https://huggingface.co/fbmacagba/bytestts-en-ljspeech) |

Each voice has its own licence and required attribution: see [Voices and licences](knowledge-base/voices-and-licences.md)
and [docs/licences.md](licences.md).

## 4. Python API

Everything public is importable from `bytestts`:

```python
from bytestts import (
    BytesTTS,
    Audio,
    VoiceInfo,
    list_voices,
    BytesTTSError,
    VoiceNotFound,
    IntegrityError,
    FrontendUnavailable,
    ConsentMissing,
    NotSupported,
    NoSpeakableText,
)
```

`import bytestts` is cheap: ONNX Runtime and the rest load on first use.

### Loading a voice

```python
BytesTTS.load_path(path, *, device="auto", threads=None) -> BytesTTS
BytesTTS.load(voice_id, *, device="auto", offline=False, threads=None) -> BytesTTS
```

- `path`: a voice package directory (a `str` or `os.PathLike`). Every file is verified first.
- `voice_id`: a catalogue voice, such as `"en-ljspeech"`. It is downloaded and verified the first time, then loaded
  from the cache. With `offline=True` nothing is downloaded: a voice that is not cached raises `VoiceNotFound`.
- `device`: `"auto"` (CUDA when it really works, else CPU), `"cpu"` or `"cuda"` (fails instead of falling back).
- `threads`: the number of CPU threads ONNX Runtime may use, `1` or more; `None` lets ONNX Runtime choose (usually
  one per physical core).

Loading takes a moment (the model is read, verified and prepared); load a voice once and reuse the object.

### Speaking

```python
tts.synthesize(text, *, speaker=None, speed=1.0) -> Audio
tts.stream(text, *, speaker=None, speed=1.0) -> Iterator[Audio]
```

- `synthesize` returns the whole text as one `Audio`, with 0.45 s of silence between sentences
  (`BytesTTS.SENTENCE_PAUSE_S`) and none at the start or end.
- `stream` yields one `Audio` per sentence (or per chunk of a very long sentence), so playback can start before the
  whole text is done. Its arguments, and the text itself, are checked when you call it, not when you start
  iterating.
- `speed` is from 0.5 (slower) to 2.0 (faster); anything else raises `ValueError`.
- `speaker` picks a speaker of a multi-speaker voice; for a single-speaker voice leave it `None`.

```python
from bytestts import BytesTTS

tts = BytesTTS.load("en-ljspeech", threads=2)
audio = tts.synthesize("The meeting moved to Friday, the 21st. Dr. Smith will chair it.", speed=0.9)
audio.save("meeting.wav")
```

### Audio

```python
Audio(samples, sample_rate)
audio.samples  # 1-D numpy float32 array, nominally in [-1, 1]
audio.sample_rate  # samples per second, e.g. 22050
audio.duration  # seconds (float)
audio.to_wav_bytes()  # 16-bit PCM mono WAV, clipped to full scale
audio.save(path)  # writes the same WAV to a file
```

`samples` can go straight to a player such as `sounddevice.play(audio.samples, audio.sample_rate)`.

### Voice information

```python
tts.info            # VoiceInfo of the loaded voice
tts.card            # the voice card (a dict) if the package has one, else None
list_voices(language=None) -> list[VoiceInfo]   # catalogue voices
```

`VoiceInfo` fields: `id`, `language`, `speakers`, `featured`, `licence`, `data_licence`, `attribution`,
`sample_rate`, `size_bytes`. `list_voices("en")` matches `en` and regional tags such as `en-gb`; voices with
featured speakers come first, then by id.

### Errors

Every error BytesTTS raises on purpose is a `BytesTTSError`:

| Exception | When |
|---|---|
| `NoSpeakableText` | The text has nothing the voice can speak (empty, only spaces or punctuation). Also a `ValueError`. |
| `VoiceNotFound` | Unknown catalogue voice or speaker, or a voice that is not cached while `offline=True`. |
| `IntegrityError` | A voice file is missing, changed, or malformed: size or SHA-256 mismatch, bad `voice.json` or `config.json`, an unsafe file name, a symbolic link, or a model that stores data in external files. |
| `FrontendUnavailable` | espeak-ng is missing, the wrong path, or fails to run. |
| `NotSupported` | The voice needs something this version lacks (an unknown engine, text cleaner or symbol layout), or `device="cuda"` cannot run on CUDA. |
| `ConsentMissing` | A fine-tuned voice without a pinned consent statement. |

Invalid arguments (`speed` out of range, `threads` below 1, an unknown `device`) raise `ValueError`.

```python
from bytestts import BytesTTS, FrontendUnavailable, IntegrityError, NoSpeakableText

try:
    tts = BytesTTS.load("en-ljspeech")
    tts.synthesize(user_text).save("out.wav")
except NoSpeakableText:
    print("Nothing to say.")
except FrontendUnavailable as e:
    print(f"espeak-ng problem: {e}")
except IntegrityError as e:
    print(f"The voice package is damaged: {e}")
```

## 5. Command line

```
bytestts --version
bytestts say [TEXT | -] [--file PATH] (--voice ID | --voice-path DIR) -o PATH|-
             [--speaker NAME] [--speed X] [--device auto|cpu|cuda] [--threads N]
bytestts voices [--language LANG]
bytestts download (ID | --all)
```

### `bytestts say`

Speaks text into a WAV file.

| Option | Meaning |
|---|---|
| `TEXT` | The text to speak. Use `-` to read UTF-8 text from standard input. |
| `--file PATH` | Read the text from a UTF-8 file instead (a BOM is fine). |
| `--voice ID` | A catalogue voice, downloaded on first use. |
| `--voice-path DIR` | A voice package directory. Exactly one of `--voice` and `--voice-path` is required. |
| `-o, --output PATH` | The WAV file to write, or `-` to write it to standard output. Required. |
| `--speaker NAME` | The speaker, for voices with several. |
| `--speed X` | 0.5 to 2.0, default 1.0. |
| `--device` | `auto` (default), `cpu` or `cuda`. |
| `--threads N` | CPU threads for synthesis, 1 or more; default: ONNX Runtime's choice. |

Give exactly one text source: `TEXT`, `-` or `--file`.

```
bytestts say "Hello there." --voice en-ljspeech -o hello.wav
bytestts say --file chapter1.txt --voice en-ljspeech --speed 1.1 -o chapter1.wav
type notes.txt | bytestts say - --voice en-ljspeech -o notes.wav        (Windows)
cat notes.txt | bytestts say - --voice en-ljspeech -o - | aplay          (Linux)
```

Text is read as bytes and decoded as UTF-8, so non-ASCII text works whatever the console's code page.

### `bytestts voices`

Lists the catalogue voices, one per line, tab-separated: id, language, number of speakers, licence, download size.
`--language en` keeps English voices only.

### `bytestts download`

Downloads a catalogue voice (or all of them with `--all`) into the cache ahead of use and prints where each one is.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Any other failure (network, files, unexpected errors) |
| 2 | Usage error, or no speakable text |
| 3 | Voice not found, failed verification, or missing consent |
| 4 | espeak-ng unavailable, or a feature not supported |
| 130 | Interrupted (Ctrl+C) |

Errors are printed as one line on standard error, never as a traceback.

## 6. How text is spoken

1. **Sentences.** The text is split into sentences with [pysbd](https://github.com/nipunsadvilkar/pySBD), in the
   voice's language when pysbd knows it and English otherwise. Long texts are split in windows, so time grows
   linearly with the length of the text.
2. **Cleaning.** Each sentence goes through the voice's own text cleaner. For English phoneme voices that expands
   numbers, currency, ordinals and years (`$5.50` becomes "five dollars fifty cents", `1999` becomes "nineteen
   ninety nine", `21st` becomes "twenty first"), common abbreviations (`Dr.` becomes "doctor", `Co.` "company")
   and symbols (`&` becomes "and"; `;` and `:` become commas, so `3:30` is read "three, thirty"). Numbers longer
   than 15 digits are read digit by digit.
3. **Phonemes or characters.** Phoneme voices send the cleaned text to espeak-ng; character voices use it as is.
4. **Tokens.** Symbols are mapped to the voice's token IDs. A character the voice has no symbol for is dropped, with
   one logged warning per character (emoji, for example). If nothing speakable remains, `NoSpeakableText` is raised.
5. **Chunks.** A sentence longer than 500 tokens is split at the space nearest its middle, repeatedly, so even a
   very long sentence without punctuation is spoken without running out of memory.
6. **Synthesis.** Each chunk is one ONNX Runtime call; the results are joined with 0.45 s pauses between sentences.

## 7. Speed, speakers, devices and threads

- **Speed.** `speed=2.0` speaks twice as fast, `0.5` half as fast. It scales the voice's own length setting, so the
  pitch does not change.
- **Speakers.** For a multi-speaker voice, `tts.info.speakers` lists the names. Without `speaker=`, the first
  featured speaker (or the first listed one) is used. Unknown names raise `VoiceNotFound`.
- **Devices.** `device="auto"` asks ONNX Runtime for CUDA when it is installed and checks which provider the
  session really runs on; when CUDA is installed but not working it logs a warning and stays on CPU.
  `device="cuda"` raises `NotSupported` with advice instead.
- **Threads.** `threads=N` caps ONNX Runtime at N CPU threads, runs operators one at a time and stops idle threads
  from spinning, which keeps the rest of the machine responsive. Leaving it `None` is fastest on a dedicated
  machine.

## 8. Voice packages and the cache

### Package layout

| File | Contents |
|---|---|
| `voice.json` | The manifest: `id`, `version`, `language`, `engine`, `sample_rate`, `licence`, `data_licence`, `attribution`, `origin`, `files` (each file's `sha256` and `size`), `speakers`, `featured` |
| `model.onnx` | The exported VITS generator |
| `config.json` | The text settings (cleaner, phonemes, symbol set) and default noise and length scales |
| `speakers.json` | Speaker names to model speaker IDs (multi-speaker voices only) |
| `card.json` | The voice card with its consent statement (fine-tuned voices only) |

`load_path` reads only files that `voice.json` lists, and refuses file names that are not plain names, symbolic
links, models stored in ONNX Runtime's own format, and models whose tensors point at files outside the model.
Packages are made with the conversion tools in `tools/convert/`; see [Design](design.md#converting-voices-maintainers).

### The cache

Downloaded catalogue voices are stored in:

- Windows: `%LOCALAPPDATA%\bytestts\Cache\voices\`
- Linux: `~/.cache/bytestts/voices/`

or under `$BYTESTTS_HOME/voices/` when `BYTESTTS_HOME` is set. Each voice gets a directory named after a digest of
its manifest, so an updated voice never mixes with an old one. Files already present and correct are reused; a damaged file is
downloaded again. Deleting the cache directory is always safe.

## 9. Configuration reference

| Environment variable | Effect |
|---|---|
| `BYTESTTS_HOME` | Where downloaded voices are cached (default: the user cache directory above). |
| `BYTESTTS_ESPEAK` | The espeak-ng executable. Default: `espeak-ng` on `PATH`. |
| `BYTESTTS_ESPEAK_DATA` | The directory that contains `espeak-ng-data`, when it is not beside the executable. |

Everything else is an argument: `device`, `threads`, `speed`, `speaker` and `offline`.

## 10. Logging

BytesTTS logs through the standard `logging` module under the `bytestts` logger and never prints from library
code. Two warnings you may see: a character the voice cannot speak was dropped, and CUDA is installed but the session
runs on CPU. To silence them:

```python
import logging

logging.getLogger("bytestts").setLevel(logging.ERROR)
```
