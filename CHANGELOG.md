# Changelog

All notable changes to BytesTTS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Security

- The control panel's server listens on `127.0.0.1` only and answers only its own page: it checks `Host`, `Origin`
  and `Sec-Fetch-Site`, requires a per-launch secret in the `Authorization` header (never a cookie), serves its
  page under a strict Content-Security-Policy, and saves through one-use tickets, so no file path passes through
  the page. The secret is never in an address: the printed address carries a one-time launch code that the page
  swaps for the secret, and a second use of that address stops the panel. Requests are not logged. See [Security model](docs/knowledge-base/security-model.md#the-control-panel).
- `load_path` reads each voice file once and uses exactly the bytes it verified. Before, a file was hashed and then
  opened again, so a file swapped on disk in between could reach ONNX Runtime unverified.
- Catalogue URLs with userinfo (`https://a@b/...`) are refused.
- A download redirect from https to a non-ASCII hostname is refused. Internationalized-name encoding could map
  fullwidth digits or ideographic full stops onto `127.0.0.1` or `localhost`. The current urllib3 already rejects
  those names, but the guard no longer depends on it.

### Fixed

- The control panel's reply to an unexpected server error carries the same `Cache-Control: no-store` and
  `X-Content-Type-Options: nosniff` headers as every other reply, and the terminal shows one line for it instead of
  a traceback.
- When the control panel's window cannot start because WebView2 is missing, the terminal shows only the panel's
  one-line reason, without pywebview's copy of the .NET error and stack trace. Other pywebview errors still print.
- A voice card that is not a JSON object is refused (`IntegrityError`) for every voice, not only fine-tuned ones.
- The CUDA advice no longer quotes an internal plan reference.
- The voice packager pins PyTorch's inter-op threads as well as its intra-op threads.
- The package ships `py.typed`, so type checkers use its annotations.

### Added

- First public version of the code, 0.1.0 (pre-release, not yet on PyPI).
- **Library:** `BytesTTS.load_path` and `BytesTTS.load`, `synthesize` and `stream` with `speed` and `speaker`,
  `Audio` with WAV output, `VoiceInfo` and `list_voices`, and an error hierarchy under `BytesTTSError`.
- **Command line:** `bytestts say`, `bytestts voices` and `bytestts download`, with stable exit codes.
- **Control panel:** `bytestts gui` opens a local panel to choose, download, repair and inspect voices, speak a
  text with speed, speaker, GPU and thread settings, play the result and save it as WAV. It opens its own window on
  Windows (WebView2, through pywebview) and the browser elsewhere, or with `--browser`; `--no-window`, `--port`
  and `--max-threads` (or `BYTESTTS_GUI_MAX_THREADS`) are the other options. It needs the new `gui` extra
  (FastAPI, uvicorn, and pywebview on Windows).
- **espeak-ng lookup:** on Windows, BytesTTS also finds espeak-ng in
  `%LOCALAPPDATA%\bytestts-tools\espeak-ng\eSpeak NG\`, where the documented `msiexec` command unpacks it, so
  `BYTESTTS_ESPEAK` is no longer needed there.
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
- **`es-css10` and `fr-css10`:** Spanish and French voices converted from Coqui TTS (trained on CSS10), checked
  against Coqui's own output by parity tests, and published in the catalogue from
  [fbmacagba/bytestts-es-css10](https://huggingface.co/fbmacagba/bytestts-es-css10) and
  [fbmacagba/bytestts-fr-css10](https://huggingface.co/fbmacagba/bytestts-fr-css10), pinned to commits `797e2a7`
  and `0e41251`. They read characters, so they need no espeak-ng. They cannot read digits: write numbers
  as words.
- **`en-vctk`:** a UK English voice with 109 speakers, converted from Coqui TTS (trained on the CSTR VCTK Corpus,
  CC BY 4.0), checked against Coqui's own output by parity tests, and published in the catalogue from
  [fbmacagba/bytestts-en-vctk](https://huggingface.co/fbmacagba/bytestts-en-vctk), pinned to commit `f62b990`.
  Ten speakers with UK accents are featured; the first, `p228`, is the default. It is the only catalogue voice
  with featured speakers, so it now lists first in `bytestts voices` and in the control panel.
- **Voice packaging:** `--rename-speaker OLD=NEW` renames a speaker without touching the model (`en-vctk`'s row 0,
  `"ED\n"` in the checkpoint, ships as `ED`), and `--featured NAME` writes the featured speakers into the
  package's `voice.json`, so a package loaded from disk defaults to the same speaker as a catalogue install.
- **Voice conversion:** Coqui voices with a language embedding and one language convert and export, with the
  language fixed inside the model; voices with several languages are refused. A speaker table is sized from the
  archive's `speaker_ids.json`.
- **Text frontend:** `multilingual_cleaners`, and typographic apostrophes (’ and ‘) read as `'` by character voices
  whose alphabet has only the ASCII one.
- **Documentation:** README, user guide, knowledge base, design notes and the voice licence register.
- **CI:** Windows and Linux, Python 3.11 to 3.14, against espeak-ng 1.52.0 built from a verified commit.
