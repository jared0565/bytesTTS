# Contributing to BytesTTS

Thanks for helping. Bug reports, documentation fixes and code are all welcome. Please open an issue before starting
a large change, so we can agree on the approach first.

## Development setup

You need [uv](https://docs.astral.sh/uv/), Python 3.11 to 3.14, and espeak-ng 1.52.0 for the phoneme tests (see
[Installing espeak-ng](docs/knowledge-base/espeak-ng.md)).

```
git clone https://github.com/jared0565/bytesTTS
cd bytesTTS
uv sync --extra train --extra server --extra torch-cpu
```

This is the same environment CI uses. It includes PyTorch (CPU build) for the conversion and export code.

## Checks

Run what CI runs before you open a pull request:

```
uv run pytest
uv run ruff check
uv run mypy
uv run python -P tools/dev/check_licences.py
```

Also format the files you change with `uv run ruff format <files>`.

- **espeak-ng tests** skip when espeak-ng is missing. Set `BYTESTTS_REQUIRE_ESPEAK=1`, as CI does, to make them fail
  instead, so a missing espeak-ng cannot pass unnoticed.
- **Optional test groups** are deselected by default: `parity` (needs a converted real voice; see
  `tests/parity/`), `quality`, `gpu` and `network`. Select one with `-m`, for example
  `uv run pytest -m parity`.
- **All Python versions on one machine:** `uv run python -P tools/dev/matrix.py` syncs a virtual environment per
  version into `.venvs/` and runs the suite in each with espeak-ng required.
- The test suite limits ONNX Runtime and the math libraries to one thread. A test that creates an ONNX Runtime
  session without `intra_op_num_threads == 1` fails on purpose. Pass `threads=1` to engines in tests.

## Code rules

- **Licence header.** Every Python file starts with `# SPDX-License-Identifier: MPL-2.0` (a test checks this).
  Code derived from Coqui TTS adds a second line: `# Derived from Coqui TTS v0.22.0, <original path>`.
- **No pickles in the runtime.** `src/` must not import `pickle`, call `torch.load`, pass `allow_pickle`, or use
  `eval`/`exec`; a test checks this. Weights are read from safetensors; models are ONNX.
- **PyTorch stays out of the runtime.** Only `bytestts.train` and `tools/` may use it; `import bytestts` must not
  pull it in.
- **Errors.** Raise the `bytestts.errors` class that fits (`IntegrityError` for a broken package, `NotSupported`
  for a missing feature, and so on), with a one-line message that says what to do. The CLI maps them to exit
  codes.
- **Dependencies.** Every new dependency must have a licence that `tools/dev/check_licences.py` accepts. Keep the
  runtime dependency list short.
- **`pyproject.toml` stays ASCII.**
- **GitHub Actions** are pinned to a full commit SHA with the version in a comment, and `run:` steps never
  interpolate event text such as titles or branch names (tests check both).
- **Style.** Ruff with a 120-character line length; strict mypy on `src/`.
- **Tests first.** New behaviour and bug fixes come with a test that fails without the change.

## Voices

New voices need a licence that allows redistribution, an entry in [docs/licences.md](docs/licences.md) and NOTICE,
and passing parity tests. Coqui checkpoints are pickles and are converted only in a sandbox; see
[Design](docs/design.md#converting-voices-maintainers). Fine-tuned voices need the speaker's consent statement in
the voice card.

## Pull requests

- Keep each pull request to one change, and write commit messages in the `type: summary` style used in the history
  (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Update the documentation and [CHANGELOG.md](CHANGELOG.md) when behaviour changes.
- CI must pass on Windows and Linux for all four Python versions.

## Maintainer tooling

`AGENTS.md`, `CLAUDE.md`, `ANTIGRAVITY.md`, `GRAPHITE.md`, `ARAMID.md`, `aramid.toml`, `.mcp.json`,
`.claude/`, `.vscode/tasks.json` and `.github/copilot-instructions.md` configure the maintainer's code-graph and
quality-gate tools and AI assistants. You do not need any of them to contribute.

## Reporting security problems

Please do not open a public issue. Follow [SECURITY.md](SECURITY.md).

By contributing, you agree that your contributions are licensed under the [Mozilla Public License 2.0](LICENSE).
