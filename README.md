# BytesTTS

[![CI](https://github.com/jared0565/bytesTTS/actions/workflows/ci.yml/badge.svg)](https://github.com/jared0565/bytesTTS/actions/workflows/ci.yml)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](LICENSE)
![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)

Text-to-speech for Python: [VITS](https://arxiv.org/abs/2106.06103) voices running on ONNX Runtime, as a library
and a command-line tool. It runs on CPU or an NVIDIA GPU, on Windows and Linux, with Python 3.11 to 3.14.
PyTorch is not needed to speak, only to convert or train voices.

```python
from bytestts import BytesTTS

tts = BytesTTS.load("en-ljspeech")  # downloaded and verified on first use
tts.synthesize("BytesTTS is working.").save("hello.wav")
```

## Features

- **Small runtime.** Synthesis needs only `onnxruntime`, `numpy`, `pysbd`, `platformdirs`, `requests` and
  `inflect`: no PyTorch, no compiler, no system packages besides espeak-ng for phoneme voices.
- **Faithful voices.** Each voice's text is prepared exactly as it was in training (its cleaner, phonemes and
  symbol set). Every catalogue voice is tested against Coqui TTS's own output: the PyTorch port matches it with
  identical durations and a signal-to-error ratio of at least 60 dB, and the ONNX model matches the port
  with identical durations and at least 50 dB.
- **Safe by default.** No pickle is ever opened at run time. Every voice file is checked against a pinned SHA-256
  before use, ONNX models that reference external files are refused, downloads are https-only on every redirect,
  and espeak-ng gets text on standard input, never on its command line.
- **Library and CLI.** `synthesize` for a whole text, `stream` for one sentence at a time, and `bytestts say` for
  files, pipes and scripts, with clear exit codes.
- **Clear licensing.** The library is MPL-2.0; every voice carries its model licence, data licence and required
  credit.

## Status

**Pre-release (0.1.0, not yet on PyPI).** The catalogue has four voices, converted from Coqui TTS and tested
against it: `en-ljspeech` (US English), `en-vctk` (UK English, 109 speakers), `es-css10` (Spanish) and `fr-css10`
(French). Each downloads on first use
from its own Hugging Face repository, pinned to a commit and checked by SHA-256. German and Italian voices are
planned. See the [changelog](CHANGELOG.md).

## Install

BytesTTS is installed from this repository:

```
pip install "bytestts @ git+https://github.com/jared0565/bytesTTS"
```

or from a clone with `pip install .`. Python 3.11 to 3.14 on Windows or Linux.

**espeak-ng.** Phoneme voices, `en-ljspeech` and `en-vctk` included, need [espeak-ng](https://github.com/espeak-ng/espeak-ng)
**1.52.0**; the Spanish and French voices read characters and do not. On Windows, without administrator rights:

```
msiexec /a espeak-ng.msi /qn TARGETDIR=%LOCALAPPDATA%\bytestts-tools\espeak-ng
```

(`espeak-ng.msi` is on the [1.52.0 release page](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0).)
BytesTTS finds it in that folder by itself. On Linux, install your distribution's `espeak-ng` package or build the
`1.52.0` tag. Details: [Installing espeak-ng](docs/knowledge-base/espeak-ng.md).

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

tts = BytesTTS.load("en-ljspeech")  # or BytesTTS.load_path(directory) for a voice package you have
audio = tts.synthesize("Hello there. How are you?", speed=1.0)
audio.save("hello.wav")  # 16-bit PCM WAV
print(audio.duration, audio.sample_rate)

for sentence in tts.stream("A longer text. One chunk per sentence."):
    print(sentence.duration)  # play or send each one as it is ready
```

Command line:

```
bytestts say "BytesTTS is working." --voice en-ljspeech -o hello.wav
bytestts say --file story.txt --voice en-ljspeech -o story.wav
echo Hello from a pipe | bytestts say - --voice en-ljspeech -o - > hello.wav
bytestts voices                      # what the catalogue offers
bytestts download en-ljspeech        # fetch ahead of time, e.g. for offline use
```

## Control panel

A local window with buttons for everything above: choose and download a voice, type or open a text, set the
speed, speaker, GPU and threads, then Speak, listen, and Save the recording as a WAV file.

```
pip install "bytestts[gui] @ git+https://github.com/jared0565/bytesTTS"
bytestts gui
```

On Windows it opens its own window (it uses the Microsoft Edge WebView2 Runtime, which Windows 11 includes); on
Linux, or with `--browser`, it opens in your web browser. It runs only on your computer: the panel listens on
`127.0.0.1` and refuses requests that do not come from its own page. See
[Control panel](docs/user-guide.md#5-control-panel) in the user guide.

## Documentation

| Document | What it covers |
|---|---|
| [User Guide](docs/user-guide.md) | Installation, the control panel, the Python API, the command line, voices, the cache, configuration |
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
| [`en-ljspeech`](https://huggingface.co/fbmacagba/bytestts-en-ljspeech) | English (US) | Apache-2.0 | Public domain |
| [`en-vctk`](https://huggingface.co/fbmacagba/bytestts-en-vctk) | English (UK) | Apache-2.0 | CC-BY-4.0 |
| [`es-css10`](https://huggingface.co/fbmacagba/bytestts-es-css10) | Spanish | BSD-3-Clause | CC0-1.0 |
| [`fr-css10`](https://huggingface.co/fbmacagba/bytestts-fr-css10) | French | BSD-3-Clause | CC0-1.0 |

`en-vctk` has 109 speakers; without `speaker=` or `--speaker`, `p228` speaks. `bytestts speakers en-vctk` lists
them, the featured ones first, and the [user guide](docs/user-guide.md#3-voices) describes the featured ones.

The Spanish and French voices cannot read digits: write numbers as words ("dos euros con cincuenta").

## Contributing and support

Bug reports and feature requests are welcome in [Issues](https://github.com/jared0565/bytesTTS/issues); please read
the [contributing guide](CONTRIBUTING.md) before opening a pull request, and report security problems privately as
described in [SECURITY.md](SECURITY.md).
