# SPDX-License-Identifier: MPL-2.0
import subprocess
import sys
import textwrap

import bytestts
from bytestts.errors import (
    BytesTTSError,
    ConsentMissing,
    FrontendUnavailable,
    IntegrityError,
    NoSpeakableText,
    NotSupported,
    VoiceNotFound,
)


def test_version() -> None:
    assert bytestts.__version__ == "0.1.0"


def test_errors_hierarchy() -> None:
    for e in (VoiceNotFound, IntegrityError, FrontendUnavailable, ConsentMissing, NotSupported, NoSpeakableText):
        assert issubclass(e, BytesTTSError)
    assert issubclass(NoSpeakableText, ValueError)


# The child walks the package itself rather than through pkgutil.walk_packages, which imports every
# package it walks into -- bytestts.train included (torch). See the Task 2 ruling in the ledger.
_CHILD = textwrap.dedent(
    """
    import importlib, pkgutil, socket, subprocess

    def boom(*args, **kwargs):
        raise RuntimeError("side effect at import time")

    subprocess.Popen = boom
    socket.socket = boom

    import bytestts

    def walk(path, prefix):
        for m in pkgutil.iter_modules(path, prefix):
            if m.name.startswith("bytestts.train"):
                continue
            mod = importlib.import_module(m.name)
            if m.ispkg:
                walk(mod.__path__, m.name + ".")

    walk(bytestts.__path__, "bytestts.")
    print("imported-ok")
    """
)


def test_import_has_no_side_effects() -> None:
    proc = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter and a constant script
        [sys.executable, "-P", "-c", _CHILD], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "imported-ok" in proc.stdout  # the child really ran to the end
