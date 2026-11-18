# Changelog

All notable changes to BytesTTS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Security

- `load_path` reads each voice file once and uses exactly the bytes it verified. Before, a file was hashed and then
  opened again, so a file swapped on disk in between could reach ONNX Runtime unverified.
- Catalogue URLs with userinfo (`https://a@b/...`) are refused.
- A download redirect from https to a non-ASCII hostname is refused. Internationalized-name encoding could map
  fullwidth digits or ideographic full stops onto `127.0.0.1` or `localhost`. The current urllib3 already rejects
  those names, but the guard no longer depends on it.

### Fixed

- A voice card that is not a JSON object is refused (`IntegrityError`) for every voice, not only fine-tuned ones.
- The CUDA advice no longer quotes an internal plan reference.
- The voice packager pins PyTorch's inter-op threads as well as its intra-op threads.
- The package ships `py.typed`, so type checkers use its annotations.

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
- **`en-ljspeech`:** converted from Coqui TTS, checked against Coqui's own output by parity tests, and published in
  the catalogue: `BytesTTS.load("en-ljspeech")` downloads it from
  [Hugging Face](https://huggingface.co/fbmacagba/bytestts-en-ljspeech), pinned to commit `8ec2c68`.
- **Documentation:** README, user guide, knowledge base, design notes and the voice licence register.
- **CI:** Windows and Linux, Python 3.11 to 3.14, against espeak-ng 1.52.0 built from a verified commit.
