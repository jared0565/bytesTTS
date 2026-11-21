# Voices and licences

## What a voice is

A BytesTTS voice is a trained [VITS](https://arxiv.org/abs/2106.06103) model exported to ONNX, plus everything
needed to feed it text exactly as it was trained: its text cleaner, whether it reads phonemes or characters, its
symbol set, and its default noise and length settings. A **voice package** holds these files:

| File | Contents |
|---|---|
| `voice.json` | The manifest: id, version, language, engine, sample rate, licences, attribution, origin, and the SHA-256 and size of every other file |
| `model.onnx` | The model |
| `config.json` | The text and synthesis settings |
| `speakers.json` | Speaker names to model speaker IDs (voices with a speaker table only) |
| `card.json` | The voice card with its consent statement (fine-tuned voices only) |

## Where voices come from

Each voice has an **origin** in its `voice.json`:

- **`catalog`**: published by BytesTTS. Today these are converted from [Coqui TTS](https://github.com/coqui-ai/TTS)
  releases: the weights are converted, not retrained.
- **`fine-tuned`**: a voice trained further on a person's recordings. These must carry a voice card with that
  person's consent statement, and BytesTTS refuses to load one without it.

### How a Coqui voice is converted (maintainers)

Coqui's releases are PyTorch pickles, which can run code when they are opened. They are therefore converted once,
by maintainers, inside an isolated sandbox with no network:

1. The release archive is pinned by URL, size and SHA-256 in `tools/convert/sources.json` before it is opened.
2. `tools/convert/convert_coqui.py` (sandbox only; it refuses to run on Windows) scans the pickle for unexpected
   code, loads it with `torch.load(weights_only=True)`, checks every weight against BytesTTS's PyTorch port of
   VITS, and writes safetensors and JSON.
3. `tools/convert/package_voice.py` reads only the safetensors and JSON, exports `model.onnx`, and writes and
   verifies the package.
4. Parity tests check the result against Coqui's own output (identical durations, signal-to-error ratio ≥ 60 dB
   for the PyTorch port; ≥ 50 dB for the ONNX model against the port).

See [Design](../design.md#converting-voices-maintainers) for the details.

## Available voices

| Voice | Language | Speakers | Model licence | Data licence | Files |
|---|---|---|---|---|---|
| `en-ljspeech` | English (US) | 1 | Apache-2.0 | Public domain | [Hugging Face](https://huggingface.co/fbmacagba/bytestts-en-ljspeech), 114.3 MB |
| `en-vctk` | English (UK) | 109 (featured: `p228`, `p232`, `p229`, `p243`, `p244`, `p287`, `p262`, `p247`, `p304`, `p253`) | Apache-2.0 | CC-BY-4.0 | [Hugging Face](https://huggingface.co/fbmacagba/bytestts-en-vctk), 121.4 MB |
| `es-css10` | Spanish | 1 | BSD-3-Clause | CC0-1.0 | [Hugging Face](https://huggingface.co/fbmacagba/bytestts-es-css10), 70.9 MB |
| `fr-css10` | French | 1 | BSD-3-Clause | CC0-1.0 | [Hugging Face](https://huggingface.co/fbmacagba/bytestts-fr-css10), 70.9 MB |

Catalogue files are downloaded from a Hugging Face commit named in BytesTTS's own catalogue, never from a branch,
so a later change to that repository cannot change what an installed BytesTTS downloads.

Planned: German and Italian voices. A voice is added to the [licence register](../licences.md)
before it is added to the catalogue. A test checks that each register row matches its voice's manifest exactly.

## Voice and speaker descriptions

The control panel and `bytestts speakers` describe voices and speakers from `details.json`, which ships inside
BytesTTS beside the catalogue and is trusted the same way. A voice's label names its language, the dataset it was
trained on and its number of speakers. The `en-vctk` speakers are described with their gender, age, accent and
region from the corpus's own `speaker-info.txt` (CC BY 4.0), kept unmodified in the BytesTTS repository under
`tools/convert/data/` and pinned by SHA-256. The descriptions change it only in form: the gender is spelled out,
accent names are split into words, `English` is shown as `English (England)`, and `Unknown` accents and the
comments in parentheses are left out. The ages are the speakers' ages when the corpus was recorded. `ED` is not
in the table. The `en-ljspeech` reader is described as female, as the CSS10 paper states of the LJ Speech
recordings.

The sample clips that ▶ Sample and ▶ Preview play are the voices' own output: one short sentence per language,
spoken by the published model for each speaker. They are in each voice's Hugging Face repository under
`samples/`, and `details.json` pins each one by commit, size and SHA-256.

## Licences

There are three separate licences to keep apart:

1. **The BytesTTS code** is licensed under the [Mozilla Public License 2.0](../../LICENSE). You can use it in open
   or closed software. If you change BytesTTS's own files and distribute them, those files stay under MPL-2.0.
2. **Each voice's model** has its own licence: Apache-2.0 for `en-ljspeech` and `en-vctk`, BSD-3-Clause for
   `es-css10` and `fr-css10`.
3. **Each voice's training data** has its own licence: public domain for `en-ljspeech` (The LJ Speech Dataset),
   CC BY 4.0 for `en-vctk` (the CSTR VCTK Corpus), and CC0-1.0 for `es-css10` and `fr-css10` (CSS10, whose
   recordings are public-domain LibriVox audiobooks).

BSD-3-Clause asks that a redistribution in binary form reproduce its copyright notice, conditions and disclaimer.
For the Spanish and French models these are in [NOTICE](../../NOTICE) and on each voice's model card.

Each voice also has an **attribution**: the credit to keep when you use or redistribute the voice. All three
strings ship in the voice's `voice.json`, and in code:

```python
from bytestts import list_voices

for v in list_voices():
    print(v.id, v.licence, v.data_licence)
    print(v.attribution)
```

For a loaded voice, use `tts.info.licence`, `tts.info.data_licence` and `tts.info.attribution`.

The full register, with sources and checksums, is [docs/licences.md](../licences.md). Third-party credits for the
code (Coqui TTS, VITS) are in [NOTICE](../../NOTICE). espeak-ng is GPL-3.0-or-later. BytesTTS runs it as a
separate program that you install yourself, and does not include it.

This page explains how the licences fit together; it is not legal advice. Read each licence for what it actually
requires.
