# How BytesTTS works

BytesTTS speaks text with [VITS](https://arxiv.org/abs/2106.06103) voices exported to ONNX and run by ONNX Runtime.
Inference needs no PyTorch: the core dependencies are `onnxruntime`, `numpy`, `pysbd`, `platformdirs`, `requests` and
`inflect`. PyTorch is used only by maintainers, to convert voices and, later, to fine-tune them (`[train]` extra).

## Goals and limits

- A small library and CLI that run the same on Windows and Linux, CPU or NVIDIA GPU, Python 3.11 to 3.14.
- Voices whose model licence and training-data licence both allow commercial use, each with its credits attached.
- No pickle is ever opened at run time, and every downloaded byte is checked against a pinned hash.
- Not goals for version 1: cloning a voice from a sample, training from scratch, or accelerators other than CUDA.

## Inference

```
text ─▶ cleaners and number expansion ─▶ sentence split (pysbd)
     ─▶ phonemes from espeak-ng over stdin, or characters ─▶ token IDs (chunks of at most 500)
     ─▶ ONNX Runtime: input, input_lengths, scales [noise, length, noise_dp], sid for multi-speaker voices
     ─▶ float32 waveform per chunk ─▶ joined with short pauses ─▶ Audio
```

- **Text frontend** (`bytestts.text`). Each voice's `config.json` says how its text was prepared when it was
  trained: the cleaner, whether it uses phonemes, and its exact symbol set. BytesTTS reproduces that, because a voice
  only sounds right on the token IDs it was trained on.
- **espeak-ng** runs as a separate process. Text always goes over standard input and never as a command-line
  argument, so text that looks like an option is spoken, not obeyed.
- **Engine** (`bytestts.engines.vits_onnx`). One `SpeechEngine` interface; a voice's `engine` field picks the runner
  from a plain mapping. The noise, length and duration-noise scales are graph inputs, so each voice's own defaults
  apply and `speed` changes the length scale. `threads` caps ONNX Runtime's CPU threads; `device="auto"` checks
  which provider the session really uses rather than trusting the list of installed ones.
- **Audio** (`bytestts.audio`). A float32 waveform plus sample rate, written as 16-bit WAV with the standard library.

## Voices

A **voice package** is a directory:

| File | What it is |
|---|---|
| `voice.json` | The manifest: id, version, language, engine, sample rate, speakers, licences, attribution, and each file's SHA-256 and size |
| `model.onnx` | The exported generator |
| `config.json` | The text frontend settings and default scales |
| `speakers.json` | Speaker names to IDs, for multi-speaker voices |

`BytesTTS.load_path` reads every file once, checks its size and SHA-256 against `voice.json`, and then uses those
exact bytes: no file is opened again after its check, so nothing can be swapped in between.

The **catalogue** (`bytestts/catalog/voices.json`) ships inside the package. Each entry is a manifest whose files
carry download URLs pinned to a repository commit (`/resolve/<40-hex commit>/`), never a branch. The downloader
checks each file's size and SHA-256, writes to a temporary file and renames it into place, uses connect and read
timeouts, follows only a few redirects and checks every hop's URL before requesting it, and never extracts
archives. Downloads are cached under the user cache directory (or `BYTESTTS_HOME`), in a directory keyed by the
manifest's digest, so an updated voice never mixes with a stale one. `offline=True` never downloads.

## Converting voices (maintainers)

Upstream VITS voices come from Coqui TTS releases as PyTorch pickles, which can run code when opened. They are
converted once, by maintainers, in an isolated sandbox with no network:

```
archive ─sha256 pin─▶ [sandbox: scan the pickle for unexpected globals ─▶ torch.load(weights_only=True)
                       ─▶ check every key and shape against the port ─▶ safetensors + JSON]
        ─▶ PyTorch port (bytestts.train.vits) loads the safetensors ─▶ ONNX export ─▶ voice package
```

- `tools/convert/sources.json` pins every upstream archive by URL, size and SHA-256 before it is opened.
- `tools/convert/convert_coqui.py` runs only in the sandbox. It refuses archives that are not pinned, pickles with
  unexpected globals, checkpoints whose weights or vocabulary do not fit the port, and voice kinds the port does not
  support yet. It writes nothing unless the whole conversion succeeds.
- `tools/convert/package_voice.py` runs on the maintainer's machine from safetensors and JSON only, exports
  `model.onnx`, and writes and verifies the package.
- The PyTorch port is derived from Coqui TTS v0.22.0 (see `NOTICE`). For `en-ljspeech`, the port reproduces Coqui
  TTS's own output on a fixed set of sentences with identical durations and a signal-to-error ratio of at least
  60 dB, and the ONNX model reproduces the port at 50 dB or better.

## Testing

- Unit tests use a tiny, randomly initialised VITS fixture, so the full engine path runs in seconds without a real
  voice.
- Parity tests compare a converted voice against reference output from Coqui TTS and against the PyTorch port. They
  need the real voice and are selected with `pytest -m parity`.
- `tests/test_licence_register.py` checks that every voice's licences and attribution in `docs/licences.md` and
  `NOTICE` match its manifest exactly.
