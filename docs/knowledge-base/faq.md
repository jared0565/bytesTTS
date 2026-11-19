# FAQ

### Where do I get a voice?

From the catalogue: `BytesTTS.load("en-ljspeech")`, or `bytestts say ... --voice en-ljspeech`, downloads the voice
on first use (114 MB, from [Hugging Face](https://huggingface.co/fbmacagba/bytestts-en-ljspeech)) and checks every
file against the SHA-256 pinned inside BytesTTS. `bytestts voices` lists what the catalogue offers. A voice package
directory you already have loads with `BytesTTS.load_path(...)` or `--voice-path`. Maintainers make packages with
the conversion tools described in [Design](../design.md#converting-voices-maintainers).

### Which languages are supported?

English (`en-ljspeech`, US English, and `en-vctk`, UK English with 109 speakers), Spanish (`es-css10`) and French
(`fr-css10`). German and Italian voices are planned.

### Why are numbers skipped in Spanish or French?

The Spanish and French voices' alphabets have no digits, and their text cleaner expands nothing, so a digit is
dropped with a warning. Write numbers as words ("dos euros con cincuenta", "deux euros cinquante").
Expanding numbers for these languages is planned. The English voice expands numbers itself.

### Do I need PyTorch?

No. Speaking needs only ONNX Runtime and a few small packages. PyTorch is needed only by the `train` extra, which is
used to convert and export voices.

### Do I need espeak-ng?

Only for voices that speak from phonemes, which includes `en-ljspeech` and `en-vctk`. The Spanish and French voices read
characters and do not need it. See [Installing espeak-ng](espeak-ng.md).
BytesTTS is tested with espeak-ng 1.52.0.

### Does it work offline?

Yes, once a voice is on disk. Download it ahead of time with `bytestts download en-ljspeech`; after that,
`BytesTTS.load("en-ljspeech")` loads it from the cache. `BytesTTS.load(voice_id, offline=True)` never downloads and
raises `VoiceNotFound` if the voice is not cached. `load_path` never touches the network.

### Does it work on macOS?

It is built and tested on Windows and Linux only. It may work on macOS, but that is not tested yet.

### Which Python versions?

3.11, 3.12, 3.13 and 3.14. CI tests all four on Windows and Linux.

### What audio format does it produce?

`Audio` holds mono float32 samples at the voice's sample rate (22,050 Hz for every catalogue voice). `save` and
`to_wav_bytes` write 16-bit PCM WAV. To get another format, convert the WAV with a tool such as ffmpeg, or pass
`audio.samples` to an audio library.

### How do I play audio without writing a file?

```python
import sounddevice as sd  # pip install sounddevice

for chunk in tts.stream("Hello there. This plays sentence by sentence."):
    sd.play(chunk.samples, chunk.sample_rate)
    sd.wait()
```

### Can I change the pitch or the voice's emotion?

No. `speed` is the only delivery setting. The voice's own noise settings come from its `config.json`.

### Can it clone a voice from a recording?

No. VITS voices speak only as the speakers they were trained on.

### Can I use it in a commercial product?

The BytesTTS code is MPL-2.0, which allows use in commercial and closed-source software, provided BytesTTS's own
files that you change stay under MPL-2.0. Each voice has its own model and data licence and an attribution to
keep: see [Voices and licences](voices-and-licences.md). `en-ljspeech` is Apache-2.0 (model) and public domain
(data); `en-vctk` is Apache-2.0 (model) and CC BY 4.0 (data); `es-css10` and `fr-css10` are BSD-3-Clause (model)
and CC0-1.0 (data). This is not legal advice.

### Is it thread-safe?

Load one `BytesTTS` object per thread or process to be safe. ONNX Runtime itself uses several threads per call
unless you set `threads`.

### Why is my CPU at 100%?

ONNX Runtime uses one thread per physical core by default. Pass `threads=1` or `2` (or `--threads` on the CLI) to
leave room for other work. See [GPU and performance](gpu-and-performance.md#cpu-threads).

### Why does BytesTTS refuse my model?

It loads only self-contained ONNX models whose every file matches the package manifest. See
[Troubleshooting](troubleshooting.md#voices) for each message and [Security model](security-model.md) for the
reasons.

### Is there a server or HTTP API?

Not yet. The `server` extra is reserved for one.

### Where is it on PyPI?

Not yet. Install from GitHub: `pip install "bytestts @ git+https://github.com/jared0565/bytesTTS"`.
