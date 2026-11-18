# Changelog

All notable changes to BytesTTS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- First public version of the code, 0.1.0 (pre-release, not yet on PyPI).
- **Library:** `BytesTTS.load_path` and `BytesTTS.load`, `synthesize` and `stream` with `speed` and `speaker`,
  `Audio` with WAV output, `VoiceInfo` and `list_voices`, and an error hierarchy under `BytesTTSError`.
- **Command line:** `bytestts say`, `bytestts voices` and `bytestts download`, with stable exit codes.
- **ONNX engine:** VITS on ONNX Runtime on CPU or CUDA, with `device="auto"` checking that CUDA really works, and
  `threads` to cap CPU use.
- **Text frontend:** sentence splitting that stays linear on long texts, English cleaners with number, currency
  and abbreviation expansion, espeak-ng phonemes over standard input, and the voice's own symbol set. Long sentences
  are split into chunks.
- **Voice packages:** strict manifests pinning every file by size and SHA-256, a refusal of external-data and ORT
  format models, consent cards for fine-tuned voices, and verified atomic downloads into a per-user cache. Every
  redirect hop is checked.
- **Conversion tools:** sandboxed conversion of Coqui TTS VITS checkpoints, a PyTorch port of the VITS generator,
  ONNX export and voice packaging.
- **`en-ljspeech`:** converted from Coqui TTS and checked against Coqui's own output by parity tests. Its files are
  not published yet, so the catalogue is empty.
- **Documentation:** README, user guide, knowledge base, design notes and the voice licence register.
- **CI:** Windows and Linux, Python 3.11 to 3.14, against espeak-ng 1.52.0 built from a verified commit.
