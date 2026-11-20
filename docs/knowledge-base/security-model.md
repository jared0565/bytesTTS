# Security model

BytesTTS downloads model files and runs a separate program (espeak-ng) on user text, and its control panel runs a
local web server. This page describes what it trusts, what it checks, and where its protection ends. To report a problem, see [SECURITY.md](../../SECURITY.md).

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

## The control panel

`bytestts gui` runs a small web server for its page. Every web page open in your browser can send requests to a
server on your own computer, so the panel checks each request before acting on it.

- **Loopback only.** The server listens on `127.0.0.1` and has no option to listen anywhere else. On Windows it
  holds its port exclusively, so another program cannot listen on the same port.
- **Its own address only.** The `Host` header must be `127.0.0.1:PORT` or `localhost:PORT`. This stops DNS
  rebinding, where a web page's own hostname is made to point at `127.0.0.1`.
- **Its own page only.** A request that carries an `Origin` header must carry the panel's own origin, and an API
  request whose `Sec-Fetch-Site` header is present must say `same-origin`. No CORS headers are sent.
- **A secret on every API call, in a header only.** Each launch makes a new random secret, and a separate one-time
  launch code that goes in the address it prints; the secret is never in an address. The page sends the code once to
  `POST /api/session`, the one API request that needs no secret (every other check here still applies to it), and
  gets the secret back in the JSON reply. It keeps the secret in `sessionStorage`, so a reload works, removes the
  code from the address bar, and sends the secret on every API request as `Authorization: Bearer`; the server
  compares the code and the secret in constant time. A second use of the same code is refused and stops the panel,
  so an address copied from browser history or another tab gives nothing but a stop. No cookie is set or read: a
  cookie would also be sent with requests from pages on other `127.0.0.1` ports. Recordings reach the player
  through `fetch` with the header, as blob URLs, never as a link an `<audio>` element could load without it.
- **Strict input.** JSON request bodies are capped at 65,536 bytes and strict: a value of the wrong type or an
  unknown field is refused. The text, speed, voice, speaker, device and threads are each checked. The one other
  body is the recording sent to `/api/export`: raw bytes under their own byte cap, accepted only as a 16-bit PCM WAV
  (mono or stereo, 8 to 48 kHz, at most 60 minutes of mono or 30 of stereo at 48 kHz), and read with Python's
  standard `wave` module, never a decoder library. No field takes a file path.
- **Saving without paths in the page.** In the window, the Save dialog is opened by Python, not by the page. The page
  gets back a random ticket that can be used once and expires after 10 minutes, and sends the recording with it. The
  server checks that the recording is a valid 16-bit PCM WAV, writes it to a temporary file beside the target, and
  renames it into place. The page never sees the folder, and error messages name only the file.
- **No bridge into Python.** The window gives the page no access to Python functions: the page talks to the server
  exactly as a browser tab does, and the same checks apply. The window runs WebView2 in private mode, so nothing is
  kept between launches, with its developer tools and reload keys turned off.
- **The page's own rules.** The page, its script and its style sheet come from the installed package only, with
  `Content-Security-Policy: default-src 'self'; media-src 'self' blob:; frame-ancestors 'none'`,
  `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`. The page has no inline script or style and
  loads nothing from the internet. File names, error messages and voice attributions are inserted as text, never as
  HTML, and links in attributions are shown as text with a Copy button, never opened.
- **Nothing private in the terminal.** The terminal shows the one-time launch address, never the secret, and for
  the app window not even the address, which the window does not need. Requests are not logged, so neither the
  secret nor your text appears in the terminal. Error replies are one line, never a traceback.
- **One CPU task at a time.** A second task is refused (the page waits for the first instead), so the panel keeps at
  most the Threads setting busy, plus one download.

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
- **The control panel guards against web pages, not against programs on your computer.** A program running under
  your account can read the launch address, just as it can read your files. If it uses the address before the
  page does, it gets the secret, but the page then finds the address already used and the panel stops, so the theft
  is not silent. In the browser, an extension allowed to read all sites can read the panel's page too.
- **Generated speech** can imitate the voice it was trained on. Fine-tuned voices therefore require a consent
  statement. Do not use voices to deceive people.
