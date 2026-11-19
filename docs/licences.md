# Voice licences

BytesTTS's code is licensed under the Mozilla Public License 2.0 (see `LICENSE`). Each voice is a separate download
with its own terms: the licence of the trained model, the licence of the recordings it was trained on, and the credit
you must keep when you use or redistribute it. The same three strings ship in each voice's `voice.json`, and
`bytestts.list_voices()` returns them as `VoiceInfo.licence`, `.data_licence` and `.attribution`.

| Voice | Model licence | Data licence | Attribution |
|---|---|---|---|
| `en-ljspeech` | Apache-2.0 | Public domain | Model: Coqui TTS tts_models/en/ljspeech/vits by Eren Gölge (Coqui), Apache-2.0; converted to ONNX by BytesTTS. Training data: The LJ Speech Dataset, Keith Ito and Linda Johnson (2017), https://keithito.com/LJ-Speech-Dataset/, public domain. |
| `es-css10` | BSD-3-Clause | CC0-1.0 | Model: Coqui TTS tts_models/es/css10/vits, Copyright Neongecko.com Inc. (@NeonGeckoCom), BSD-3-Clause; converted to ONNX by BytesTTS. Training data: CSS10 Spanish, Kyubyong Park and Thomas Mulc (2019), CC0-1.0; audio from LibriVox (reader: Tux), public domain. |
| `fr-css10` | BSD-3-Clause | CC0-1.0 | Model: Coqui TTS tts_models/fr/css10/vits, Copyright Neongecko.com Inc. (@NeonGeckoCom), BSD-3-Clause; converted to ONNX by BytesTTS. Training data: CSS10 French, Kyubyong Park and Thomas Mulc (2019), CC0-1.0; audio from LibriVox (reader: Gilles G. Le Blanc), public domain. |
| `en-vctk` | Apache-2.0 | CC-BY-4.0 | Model: Coqui TTS tts_models/en/vctk/vits by Eren Gölge (Coqui), Apache-2.0; converted to ONNX by BytesTTS. Training data: CSTR VCTK Corpus (version 0.92), Junichi Yamagishi, Christophe Veaux and Kirsten MacDonald, The Centre for Speech Technology Research, University of Edinburgh (2019), https://doi.org/10.7488/ds/2645, CC BY 4.0. |

## Sources

- `en-ljspeech`
  - Model: Coqui TTS release `v0.6.1_models`, `tts_models--en--ljspeech--vits.zip`
    (sha256 `061ed8902f57c2dc8821b675bd15b7134cec2fb5d576dafcb928b6f5bf1a3489`), licence "apache 2.0" in
    Coqui's model list. BytesTTS converts the weights to ONNX; it does not retrain them.
  - Data: [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/), whose page gives its licence as Public
    Domain. The recordings come from LibriVox, the texts from works published between 1884 and 1964.
  - Published files: [fbmacagba/bytestts-en-ljspeech](https://huggingface.co/fbmacagba/bytestts-en-ljspeech) on
    Hugging Face; the catalogue pins commit `8ec2c68f8187a2d68fbf97c657b2ccf4f73fbfb0`.

- `es-css10` and `fr-css10`
  - Model: Coqui TTS release `v0.8.0_models`, `tts_models--es--css10--vits.zip`
    (sha256 `3b15b180b52dc0884934e81be2935a23bd7c2be58410c158c4d40e7deba1807f`) and
    `tts_models--fr--css10--vits.zip` (sha256 `e946732aaef60b3e6462064b5f0e0286bc48ffa324474d0bde578ba60e566219`).
    Coqui's model list gives the author as `@NeonGeckoCom` and the licence as "bsd-3-clause"; so do Neon's own
    repositories, [neongeckocom/tts-vits-css10-es](https://huggingface.co/neongeckocom/tts-vits-css10-es) and
    [neongeckocom/tts-vits-css10-fr](https://huggingface.co/neongeckocom/tts-vits-css10-fr). Neither ships a
    copyright line; the holder named here, Neongecko.com Inc., is the one in Neon's own BSD-3-Clause licence. The
    licence's conditions and disclaimer are in `NOTICE` and in each voice's model card. BytesTTS converts the weights
    to ONNX; it does not retrain them.
  - Data: CSS10 (Kyubyong Park and Thomas Mulc, "CSS10: A Collection of Single Speaker Speech Datasets for 10
    Languages", Interspeech 2019). The
    [Spanish](https://www.kaggle.com/datasets/bryanpark/spanish-single-speaker-speech-dataset) and
    [French](https://www.kaggle.com/datasets/bryanpark/french-single-speaker-speech-dataset) datasets are published
    under "CC0: Public Domain". Their recordings are LibriVox audiobooks, in the public domain, read by Tux
    (Spanish) and Gilles G. Le Blanc (French).

- `en-vctk`
  - Model: Coqui TTS release `v0.6.1_models`, `tts_models--en--vctk--vits.zip`
    (sha256 `ad753e5200614907627495b6177a08d755df1dbbde30eb6e31264caa2a3f3eaa`). Coqui's model list gives the
    author as "Eren @erogol" and the licence as "apache 2.0". BytesTTS converts the weights to ONNX; it does not
    retrain them.
  - Data: [CSTR VCTK Corpus, version 0.92](https://doi.org/10.7488/ds/2645) (Junichi Yamagishi, Christophe Veaux and
    Kirsten MacDonald, The Centre for Speech Technology Research, University of Edinburgh, 2019), on Edinburgh
    DataShare. The record's `license_text.txt` is the legal code of
    [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). BytesTTS does not redistribute the recordings.
  - Speakers: the model has 109 speaker rows. 108 are VCTK speakers, named by their VCTK ids (`p225` to `p376`).
    Row 0 is named `"ED\n"` in the checkpoint and is not a VCTK speaker id; the package names it `ED`. The accents
    shown for the featured speakers come from the corpus's own `speaker-info.txt`.
  - Published files: [fbmacagba/bytestts-en-vctk](https://huggingface.co/fbmacagba/bytestts-en-vctk) on Hugging
    Face; the catalogue pins commit `f62b990f0bd6992cb860f303e1f607f18d4b0c13`.

A voice is added here before it is added to the catalogue; `tests/test_licence_register.py` checks that every row
matches its voice's manifest exactly.

## Software used by the control panel

On Windows, `pip install "bytestts[gui]"` also installs the packages that draw the control panel's window. BytesTTS
does not redistribute them; each keeps its own licence:

- **pywebview** (BSD-3-Clause, Copyright Roman Sirokov). Its wheel bundles the Microsoft WebView2 SDK 1.0.3856.49
  (`Microsoft.Web.WebView2.Core.dll`, `Microsoft.Web.WebView2.WinForms.dll` and `WebView2Loader.dll`), Copyright
  Microsoft Corporation, under a BSD-3-Clause-style licence. The SDK's NOTICE names Antlr3.Runtime and
  StringTemplate4, both BSD-3-Clause.
- **pythonnet** (MIT), with its `Python.Runtime.dll`.
- **clr_loader** (MIT, Copyright Benedikt Reinartz), with its `ClrLoader.dll`. Its package metadata names no
  licence, so the licence check lists it as reviewed by hand at version 0.3.1.
