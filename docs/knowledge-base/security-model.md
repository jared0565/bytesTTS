# Security model

BytesTTS downloads model files and runs a separate program (espeak-ng) on user text. This page describes what it
trusts, what it checks, and where its protection ends. To report a problem, see [SECURITY.md](../../SECURITY.md).

## What is trusted

- **The BytesTTS package you installed**, including the catalogue `voices.json` it ships with. The catalogue pins
  every voice file by size and SHA-256, so a download is trusted only after it matches.
- **The voice package directories you load with `load_path`.** BytesTTS verifies them against their own
  `voice.json`, which proves the files are the ones the manifest names. It does not prove who made the package, so
  load packages only from sources you trust.
- **espeak-ng**, as installed by you.

## No pickles at run time

PyTorch checkpoints are pickles, which can run code when they are opened. BytesTTS never opens one when it loads or
speaks a voice: voices are ONNX models, and weights in the conversion tools are read from safetensors. Coqui's
original pickles are opened only by maintainers, inside a sandbox with no network, after their archive matches a
pinned SHA-256, a scan for unexpected code, and `torch.load(weights_only=True)`.

## Voice packages

Before anything in a package is used:

- `voice.json` is parsed strictly: valid UTF-8 JSON, at most 1 MiB, no duplicate keys, no unknown keys, and every
  field of the right type.
- Only files listed in `voice.json` are read. Their names must be plain (1 to 128 characters from `A-Z a-z 0-9 . _ -`)
  so no name can reach outside the directory, and symbolic links are refused.
- Every file is read once and must match its pinned size and SHA-256. Those exact bytes are then used: no file is
  opened again after its check, so a file swapped on disk in between cannot reach ONNX Runtime or the text
  frontend.
- A voice card, if listed, must be a JSON object. A fine-tuned voice must carry a pinned card with a consent
  statement.

## ONNX models

An ONNX model can store its weights as **external data**, naming files that ONNX Runtime then opens relative to the
model, which could be anywhere. BytesTTS parses the model itself before ONNX Runtime sees it and refuses any model in
which a tensor, at any depth, uses external data. It also refuses ONNX Runtime's own `.ort` format. The session is
built from the verified bytes, with ONNX Runtime's external-data directory pointed at an empty private directory.

## espeak-ng

- User text goes to espeak-ng on **standard input** only. The command line holds fixed options and the voice's
  language, which must match a language-tag pattern, so no text can become an option.
- espeak-ng is found through `BYTESTTS_ESPEAK` or the **absolute** entries of `PATH` only. The current directory
  and relative `PATH` entries are skipped, so a file named `espeak-ng` in a downloaded folder is never run. On Windows
  only `espeak-ng.exe` is accepted, never a `.bat` or `.cmd` (those run through `cmd.exe`).
- Each call has a 30-second time limit.

## Downloads

- Each file goes to a unique temporary `.part` file, is checked against its pinned size and SHA-256, and only then
  is renamed into place. On any failure the `.part` file is deleted. A server announcing or sending more bytes than
  the pinned size is cut off.
- Redirects are followed by hand so that every hop is checked before it is requested. Downloads are https only,
  except plain http to `127.0.0.1` or `localhost` for local mirrors and tests. A hop that came from https may never
  go back to http, nor reach `localhost` (with or without a trailing dot) or a non-global IP address, including
  IPv6 and the short, octal, hexadecimal and single-number IPv4 forms (`127.1`, `0177.0.0.1`, `0x7f000001`,
  `2130706433`). Such a hop's hostname must also be plain ASCII: internationalized-name encoding maps fullwidth
  digits and ideographic full stops onto ASCII, so `１２７。０。０。１` could otherwise spell `127.0.0.1`. The cost is that
  a redirect to a public internationalized name is refused too.
- Archives are never extracted.

## Known limits

- **DNS.** A redirect to an https *hostname* that resolves to an internal address is not blocked; that would need
  DNS pinning. TLS still requires the host to present a valid certificate for that name.
- **Consent is declared, not proven.** A package states its own origin, and only `fine-tuned` packages must carry
  a consent statement. Whoever builds a package can mislabel it, or write a statement nobody made, and BytesTTS
  cannot tell. The check is a rule for the voices BytesTTS publishes, not a barrier against a dishonest packager.
- **Your own files.** Keep voice directories and the cache writable only by your own account: anyone who can write
  there can replace a whole package, manifest included.
- **Authenticity.** `load_path` verifies a package against its own manifest, so a package made by someone else is
  internally consistent, not endorsed. Catalogue voices are checked against the pins shipped inside BytesTTS.
- **espeak-ng itself** processes your text. Keep it up to date from its official releases.
- **Generated speech** can imitate the voice it was trained on. Fine-tuned voices therefore require a consent
  statement. Do not use voices to deceive people.
