# Coqui reference data (Task 16 tests)

- `en-ljspeech.config.json`: the `config.json` member of Coqui TTS's
  `tts_models--en--ljspeech--vits.zip` (release `v0.6.1_models`; archive sha256
  `061ed8902f57c2dc8821b675bd15b7134cec2fb5d576dafcb928b6f5bf1a3489`), as Phase 0's fetch tool recorded it:
  parsed and re-written with `json.dumps(indent=1)` (LF line ends), so the same content but not the member's bytes (4,630 vs 5,734).
  Coqui publishes this model under the Apache License 2.0. The file holds configuration only, no weights.
- `en-ljspeech.voice_config.json`: the `voice_config` block that Phase 0 (P0.7) recorded from TTS-dev for the same
  voice. `tools/convert/coqui_keys.py:extract_voice_config` must reproduce it exactly.
