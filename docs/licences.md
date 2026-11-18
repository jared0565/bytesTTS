# Voice licences

BytesTTS's code is licensed under the Mozilla Public License 2.0 (see `LICENSE`). Each voice is a separate download
with its own terms: the licence of the trained model, the licence of the recordings it was trained on, and the credit
you must keep when you use or redistribute it. The same three strings ship in each voice's `voice.json`, and
`bytestts.list_voices()` returns them as `VoiceInfo.licence`, `.data_licence` and `.attribution`.

| Voice | Model licence | Data licence | Attribution |
|---|---|---|---|
| `en-ljspeech` | Apache-2.0 | Public domain | Model: Coqui TTS tts_models/en/ljspeech/vits by Eren Gölge (Coqui), Apache-2.0; converted to ONNX by BytesTTS. Training data: The LJ Speech Dataset, Keith Ito and Linda Johnson (2017), https://keithito.com/LJ-Speech-Dataset/, public domain. |

## Sources

- `en-ljspeech`
  - Model: Coqui TTS release `v0.6.1_models`, `tts_models--en--ljspeech--vits.zip`
    (sha256 `061ed8902f57c2dc8821b675bd15b7134cec2fb5d576dafcb928b6f5bf1a3489`), licence "apache 2.0" in
    Coqui's model list. BytesTTS converts the weights to ONNX; it does not retrain them.
  - Data: [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/), whose page gives its licence as Public
    Domain. The recordings come from LibriVox, the texts from works published between 1884 and 1964.
  - Published files: [fbmacagba/bytestts-en-ljspeech](https://huggingface.co/fbmacagba/bytestts-en-ljspeech) on
    Hugging Face; the catalogue pins commit `8ec2c68f8187a2d68fbf97c657b2ccf4f73fbfb0`.

A voice is added here before it is added to the catalogue; `tests/test_licence_register.py` checks that every row
matches its voice's manifest exactly.
