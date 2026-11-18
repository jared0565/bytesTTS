# Installing espeak-ng

Voices trained on phonemes, such as `en-ljspeech`, need [espeak-ng](https://github.com/espeak-ng/espeak-ng) to turn
text into phonemes. Voices trained on plain characters do not need it. BytesTTS runs espeak-ng as a separate program;
it does not link to it or bundle it, so you install it yourself (espeak-ng is licensed GPL-3.0-or-later).

## Which version

BytesTTS is tested with **espeak-ng 1.52.0**, and its CI uses exactly that release. Other versions usually work but
may pronounce some words differently, which changes what the voice says. Check yours with `espeak-ng --version`; the
first line ends with the version, for example `eSpeak NG text-to-speech: 1.52.0`.

## Windows

No administrator rights are needed.

1. Download `espeak-ng.msi` from the [1.52.0 release](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0).
2. Unpack it into your profile instead of installing it:

   ```
   msiexec /a espeak-ng.msi /qn TARGETDIR=%LOCALAPPDATA%\bytestts-tools\espeak-ng
   ```

   This creates `%LOCALAPPDATA%\bytestts-tools\espeak-ng\eSpeak NG\` with `espeak-ng.exe` and `espeak-ng-data\`.
   BytesTTS looks in that folder by itself, so nothing else is needed.
3. Check it:

   ```
   "%LOCALAPPDATA%\bytestts-tools\espeak-ng\eSpeak NG\espeak-ng.exe" --version
   ```

If you unpacked it into another folder, tell BytesTTS where it is:

```
setx BYTESTTS_ESPEAK "C:\path\to\eSpeak NG\espeak-ng.exe"
```

`setx` applies to new windows; use `set` instead for the current one only.

If you install espeak-ng with the normal installer instead, its folder is usually added to `PATH` and BytesTTS finds
it there.

## Linux

The simplest route is your distribution's package:

```
sudo apt install espeak-ng          # Debian, Ubuntu
sudo dnf install espeak-ng          # Fedora
```

Distributions often ship an older release (Ubuntu 24.04 has 1.51). For exactly 1.52.0, build it from source into
your home directory; this is what BytesTTS's CI does:

```
sudo apt install build-essential autoconf automake libtool pkg-config git
git clone --depth 1 --branch 1.52.0 https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
./autogen.sh
./configure --prefix="$HOME/espeak-ng"
make
make install
export BYTESTTS_ESPEAK="$HOME/espeak-ng/bin/espeak-ng"
export BYTESTTS_ESPEAK_DATA="$HOME/espeak-ng/share"
export LD_LIBRARY_PATH="$HOME/espeak-ng/lib"
```

Add the three `export` lines to your shell profile to keep them.

## How BytesTTS finds espeak-ng

1. `BYTESTTS_ESPEAK`, if set. It must name an existing file, or loading a phoneme voice fails with
   "espeak-ng executable not found at ...".
2. Otherwise `espeak-ng` on `PATH`, searching **absolute `PATH` entries only**. The current directory, and empty or
   relative `PATH` entries, are never searched, so an `espeak-ng` file planted in a downloaded folder cannot run. On
   Windows only `espeak-ng.exe` counts (never a `.bat` or `.cmd`); on Linux the file must be executable.
3. Otherwise, on Windows only, `%LOCALAPPDATA%\bytestts-tools\espeak-ng\eSpeak NG\espeak-ng.exe`, the folder the
   steps above unpack into. `LOCALAPPDATA` must be an absolute path, so an unset or relative value never points the
   lookup at the current directory.

Its data directory (`espeak-ng-data`) is taken from:

1. `BYTESTTS_ESPEAK_DATA`, if set: the directory that **contains** `espeak-ng-data`, not `espeak-ng-data` itself.
2. Otherwise the executable's own folder, if `espeak-ng-data` is there (the Windows layout above).
3. Otherwise espeak-ng's built-in default (normal for distribution packages).

BytesTTS checks that the data directory exists before running espeak-ng: espeak-ng 1.52.0 on Windows crashes
(exit code `3221225477`, `0xC0000005`) instead of reporting a wrong data path.

## How text reaches espeak-ng

Text goes to espeak-ng on **standard input**, as UTF-8, never on its command line. The command line holds only fixed
options and the voice's language name, which must look like a language tag (`en-us`, `de`). So text that starts
with `-` is spoken rather than read as an option, and non-ASCII text such as `café naïve` arrives intact whatever the
Windows console code page. Each call has a 30-second time limit.

## Common problems

| Message | Fix |
|---|---|
| `espeak-ng not found.` | Install espeak-ng: on Windows, unpack it into the folder above; elsewhere set `BYTESTTS_ESPEAK`, or put its folder on `PATH` as an absolute path. |
| `espeak-ng executable not found at ...` | `BYTESTTS_ESPEAK` points at a file that does not exist; check the path and quotes. |
| `no espeak-ng-data under ...` | `BYTESTTS_ESPEAK_DATA` must be the folder that contains `espeak-ng-data`. |
| `espeak-ng failed (exit code ...)` | Run `"%BYTESTTS_ESPEAK%" --version` (or `espeak-ng --version`) by hand; on Linux with a custom prefix, set `LD_LIBRARY_PATH`. |
| `espeak-ng could not run: ...` | The file is not a working executable for this system, or a call took more than 30 seconds. |
