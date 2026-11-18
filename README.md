# BytesTTS

[![CI](https://github.com/jared0565/bytesTTS/actions/workflows/ci.yml/badge.svg)](https://github.com/jared0565/bytesTTS/actions/workflows/ci.yml)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](LICENSE)
![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)

Text-to-speech for Python: [VITS](https://arxiv.org/abs/2106.06103) voices running on ONNX Runtime, as a library
and a command-line tool. It runs on CPU or an NVIDIA GPU, on Windows and Linux, with Python 3.11 to 3.14.
PyTorch is not needed to speak, only to convert or train voices.

```python
from bytestts import BytesTTS

tts = BytesTTS.load_path("voices/en-ljspeech")
tts.synthesize("BytesTTS is working.").save("hello.wav")
```

## Features

- **Small runtime.** Synthesis needs only `onnxruntime`, `numpy`, `pysbd`, `platformdirs`, `requests` and
  `inflect`: no PyTorch, no compiler, no system packages besides espeak-ng for phoneme voices.
- **Faithful voices.** Each voice's text is prepared exactly as it was in training (its cleaner, phonemes and
  symbol set). The converted `en-ljspeech` voice is tested against Coqui TTS's own output: the PyTorch port matches
  it with identical durations and a signal-to-error ratio of at least 60 dB, and the ONNX model matches the port
  with identical durations and at least 50 dB.
- **Safe by default.** No pickle is ever opened at run time. Every voice file is checked against a pinned SHA-256
  before use, ONNX models that reference external files are refused, downloads are https-only on every redirect,
  and espeak-ng gets text on standard input, never on its command line.
- **Library and CLI.** `synthesize` for a whole text, `stream` for one sentence at a time, and `bytestts say` for
  files, pipes and scripts, with clear exit codes.
- **Clear licensing.** The library is MPL-2.0; every voice carries its model licence, data licence and required
  credit.

## Status

**Pre-release (0.1.0, not yet on PyPI).** One English voice, `en-ljspeech`, is converted from Coqui TTS and
verified against it. The downloadable voice catalogue opens once the voice files are published; until then a voice
is loaded from a voice package directory with `BytesTTS.load_path` or `bytestts say --voice-path`. German,
Spanish, French and Italian voices are planned. See the [changelog](CHANGELOG.md).

## Install

BytesTTS is installed from this repository:

```
pip install "bytestts @ git+https://github.com/jared0565/bytesTTS"
```

or from a clone with `pip install .`. Python 3.11 to 3.14 on Windows or Linux.

**espeak-ng.** Phoneme voices, `en-ljspeech` included, need [espeak-ng](https://github.com/espeak-ng/espeak-ng)
**1.52.0**. On Windows, without administrator rights:

```
msiexec /a espeak-ng.msi /qn TARGETDIR=%LOCALAPPDATA%\bytestts-tools\espeak-ng
set BYTESTTS_ESPEAK=%LOCALAPPDATA%\bytestts-tools\espeak-ng\eSpeak NG\espeak-ng.exe
```

(`espeak-ng.msi` is on the [1.52.0 release page](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0).) On
Linux, install your distribution's `espeak-ng` package or build the `1.52.0` tag. Details:
[Installing espeak-ng](docs/knowledge-base/espeak-ng.md).

**NVIDIA GPU (optional).** Install the CUDA 13 runtime and cuDNN 9 from NVIDIA, then:

```
pip install "bytestts[gpu] @ git+https://github.com/jared0565/bytesTTS"
pip install --force-reinstall --no-deps onnxruntime-gpu
```

See [GPU and performance](docs/knowledge-base/gpu-and-performance.md) for why the second line is needed.

## Quick start

Python:

```python
from bytestts import BytesTTS

tts = BytesTTS.load_path("path/to/en-ljspeech")      # a voice package directory
audio = tts.synthesize("Hello there. How are you?", speed=1.0)
audio.save("hello.wav")                               # 16-bit PCM WAV
print(audio.duration, audio.sample_rate)

for sentence in tts.stream("A longer text. One chunk per sentence."):
    print(sentence.duration)                          # play or send each one as it is ready
```

Command line:

```
bytestts say "BytesTTS is working." --voice-path path/to/en-ljspeech -o hello.wav
bytestts say --file story.txt --voice-path path/to/en-ljspeech -o story.wav
echo Hello from a pipe | bytestts say - --voice-path path/to/en-ljspeech -o - > hello.wav
```

## Documentation

| Document | What it covers |
|---|---|
| [User Guide](docs/user-guide.md) | Installation, the Python API, the command line, voices, the cache, configuration |
| [Knowledge Base](docs/knowledge-base/README.md) | Troubleshooting, espeak-ng, GPU and performance, voices and licences, the security model, FAQ |
| [Design](docs/design.md) | How BytesTTS works inside: the text frontend, the ONNX engine, voice packages, voice conversion |
| [Voice licences](docs/licences.md) | Every voice's model licence, data licence and attribution |
| [Contributing](CONTRIBUTING.md) | Development setup, tests, code rules, pull requests |
| [Security](SECURITY.md) | How to report a vulnerability |
| [Changelog](CHANGELOG.md) | What changed in each version |

## Voices and licences

The library is licensed under the [Mozilla Public License 2.0](LICENSE). Each voice is a separate download with its
own model licence, data licence and required credit, listed in [docs/licences.md](docs/licences.md) and available
in code as `VoiceInfo.licence`, `.data_licence` and `.attribution`. Keep the attribution when you use or
redistribute a voice. Third-party credits are in [NOTICE](NOTICE).

| Voice | Language | Model licence | Data licence |
|---|---|---|---|
| `en-ljspeech` | English (US) | Apache-2.0 | Public domain |

## Contributing and support

Bug reports and feature requests are welcome in [Issues](https://github.com/jared0565/bytesTTS/issues); please read
the [contributing guide](CONTRIBUTING.md) before opening a pull request, and report security problems privately as
described in [SECURITY.md](SECURITY.md).
