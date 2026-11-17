# SPDX-License-Identifier: MPL-2.0
"""Local test matrix: Python 3.11-3.14, each in its own per-OS venv, with espeak-ng required.

    uv run python -P tools/dev/matrix.py

For each version: sync with the CI command into .venvs/<platform>-py<version>, then run pytest
with BYTESTTS_REQUIRE_ESPEAK=1. Exits non-zero if any version fails or is not installed locally.
On this PC, run only with the operator's OK: the suite builds ONNX sessions (plan, This PC).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

VERSIONS = ("3.11", "3.12", "3.13", "3.14")
SYNC = ["uv", "sync", "--extra", "train", "--extra", "server", "--extra", "torch-cpu"]
ROOT = Path(__file__).resolve().parents[2]


def espeak_version() -> str:
    """The espeak-ng the tests will find: $BYTESTTS_ESPEAK, else espeak-ng on PATH."""
    exe = os.environ.get("BYTESTTS_ESPEAK") or shutil.which("espeak-ng")
    if not exe:
        return "not found"
    env = dict(os.environ)
    if os.environ.get("BYTESTTS_ESPEAK_DATA"):
        env["ESPEAK_DATA_PATH"] = os.environ["BYTESTTS_ESPEAK_DATA"]
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, env=env, timeout=30)  # noqa: S603
    except OSError as e:
        return f"error: {e}"
    words = out.stdout.split()
    return words[3] if out.returncode == 0 and len(words) > 3 else f"error (exit {out.returncode})"


def run_version(version: str) -> str:
    """Sync and test one Python version. Returns "pass", "FAIL (...)" or "missing"."""
    found = subprocess.run(["uv", "python", "find", version], capture_output=True, text=True)  # noqa: S603, S607
    if found.returncode != 0:
        return "missing"
    env = dict(os.environ)
    env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venvs" / f"{sys.platform}-py{version}")
    env["BYTESTTS_REQUIRE_ESPEAK"] = "1"
    sync = subprocess.run([*SYNC, "--python", version], cwd=ROOT, env=env)  # noqa: S603
    if sync.returncode != 0:
        return f"FAIL (sync exit {sync.returncode})"
    test = subprocess.run(["uv", "run", "--python", version, "pytest"], cwd=ROOT, env=env)  # noqa: S603, S607
    return "pass" if test.returncode == 0 else f"FAIL (pytest exit {test.returncode})"


def main() -> int:
    espeak = espeak_version()
    results = {v: run_version(v) for v in VERSIONS}
    print(f"\n{'python':<8} {'result':<24} espeak-ng")
    for v, r in results.items():
        print(f"{v:<8} {r:<24} {espeak}")
    return 0 if all(r == "pass" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
