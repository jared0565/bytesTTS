## What and why

<!-- What this changes and why. Link the issue: "Fixes #123". -->

## Checks

- [ ] `uv run pytest` passes (with `BYTESTTS_REQUIRE_ESPEAK=1` if espeak-ng is installed)
- [ ] `uv run ruff check` and `uv run mypy` pass; changed files are formatted with `uv run ruff format`
- [ ] `uv run python -P tools/dev/check_licences.py` passes (new dependencies only)
- [ ] New behaviour or fixes come with a test that fails without the change
- [ ] New Python files start with `# SPDX-License-Identifier: MPL-2.0`
- [ ] Documentation and `CHANGELOG.md` are updated where behaviour changed
