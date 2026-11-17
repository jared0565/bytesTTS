# BytesTTS

Text-to-speech for Python: VITS voices on ONNX Runtime, as a library and a command-line tool. It runs on CPU or an
NVIDIA GPU, on Windows and Linux, with Python 3.11 to 3.14. PyTorch is not needed to speak, only to convert or
train voices.

**Status: pre-release.** One English voice, `en-ljspeech`, is converted from Coqui TTS and verified against it.
The downloadable voice catalogue opens once the voice files are published; until then a voice is loaded from a
voice package directory (`--voice-path`, `BytesTTS.load_path`). BytesTTS is not on PyPI yet.

## Install

From a clone of this repository:

```
pip install .
```

### CPU or GPU

The default install runs on CPU through `onnxruntime`. For an NVIDIA GPU:

1. Install the CUDA 13 runtime and cuDNN 9 for CUDA 13 from NVIDIA. BytesTTS does not bundle NVIDIA's libraries.
2. Install the `gpu` extra, then reinstall `onnxruntime-gpu` so its files are the ones on disk (`onnxruntime` and
   `onnxruntime-gpu` share files, and whichever is installed last wins):

   ```
   pip install ".[gpu]"
   pip install --force-reinstall --no-deps onnxruntime-gpu
   ```

`device="auto"` (the default) uses CUDA when ONNX Runtime can actually run on it, and otherwise CPU, with a warning
when CUDA is installed but cannot run. `device="cuda"` fails with advice instead of falling back.

### espeak-ng

Voices that speak from phonemes, `en-ljspeech` included, need [espeak-ng](https://github.com/espeak-ng/espeak-ng).
BytesTTS runs it as a separate program and passes text over standard input; it does not link to it. BytesTTS is
tested with espeak-ng **1.52.0**; other versions may pronounce some words differently.

- **Windows** (no administrator rights needed): download `espeak-ng.msi` from the
  [1.52.0 release](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0), unpack it, and point BytesTTS at it:

  ```
  msiexec /a espeak-ng.msi /qn TARGETDIR=%LOCALAPPDATA%\bytestts-tools\espeak-ng
  set BYTESTTS_ESPEAK=%LOCALAPPDATA%\bytestts-tools\espeak-ng\eSpeak NG\espeak-ng.exe
  ```

- **Linux:** install your distribution's `espeak-ng` package, or build the `1.52.0` tag from source.

BytesTTS looks for espeak-ng in `BYTESTTS_ESPEAK`, then on `PATH`. If its data is not next to the executable, set
`BYTESTTS_ESPEAK_DATA` to the directory that contains `espeak-ng-data`.

## Quick start

Python:

```python
from bytestts import BytesTTS

tts = BytesTTS.load_path("path/to/en-ljspeech")  # a voice package directory
audio = tts.synthesize("BytesTTS is working.", speed=1.0)
audio.save("hello.wav")

for chunk in tts.stream("A longer text. One chunk per sentence."):
    print(chunk.duration)
```

Command line:

```
bytestts say "BytesTTS is working." --voice-path path/to/en-ljspeech -o hello.wav
bytestts say --file story.txt --voice-path path/to/en-ljspeech -o - > story.wav
```

Once the catalogue is published, `--voice ID` and `BytesTTS.load(ID)` download a voice on first use and verify every
file against its pinned SHA-256; `bytestts voices` lists the catalogue and `bytestts download` fetches voices ahead
of time.

Useful options:

- `speed` (0.5 to 2.0) and, for multi-speaker voices, `speaker`.
- `threads` (`--threads N`, or `threads=` on `load`/`load_path`) caps the CPU threads ONNX Runtime uses. The
  default lets ONNX Runtime choose, which is usually one thread per physical core.
- `BYTESTTS_HOME` moves the voice cache.

## Voices and licences

The library is licensed under the [Mozilla Public License 2.0](LICENSE). Each voice is a separate download with its
own model licence, data licence and required credit, listed in [docs/licences.md](docs/licences.md) and returned by
`list_voices()` as `VoiceInfo.licence`, `.data_licence` and `.attribution`. Keep the attribution when you use or
redistribute a voice. Third-party credits are in [NOTICE](NOTICE).

## More

[docs/design.md](docs/design.md) describes how BytesTTS works: the text frontend, the ONNX engine, voice packages
and how voices are converted.
