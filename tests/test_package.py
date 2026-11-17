# SPDX-License-Identifier: MPL-2.0
import subprocess
import sys
import textwrap

import pytest

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

    # Subclasses, not functions: modules may subclass these at import time (ssl does `class SSLSocket(socket)`,
    # reached through requests), which is not a side effect. Creating one is.
    class NoSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            boom()

    class NoPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            boom()

    subprocess.Popen = NoPopen
    socket.socket = NoSocket

    import bytestts
    # PROBE

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


def _run_child(probe: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- fixed argv: this interpreter and a constant script
        [sys.executable, "-P", "-c", _CHILD.replace("# PROBE", probe)], capture_output=True, text=True, timeout=120
    )


def test_import_has_no_side_effects() -> None:
    proc = _run_child()
    assert proc.returncode == 0, proc.stderr
    assert "imported-ok" in proc.stdout  # the child really ran to the end


@pytest.mark.parametrize(
    "probe",
    [
        "socket.socket()",
        "socket.create_connection(('127.0.0.1', 9), timeout=1)",
        "subprocess.run(['cmd'])",
        "import ssl; ssl.create_default_context().wrap_socket(socket.socket())",
    ],
)
def test_side_effect_guard_fires(probe: str) -> None:  # the control: the guard above can fail
    proc = _run_child(probe)
    assert proc.returncode != 0 and "side effect at import time" in proc.stderr
    assert "imported-ok" not in proc.stdout


# The converter runs in the sandbox's reference environments, which have torch but not the runtime's
# dependencies (onnxruntime, requests, platformdirs, pysbd). Its imports must not need them.
_NO_RUNTIME = textwrap.dedent(
    """
    import sys
    for name in ("onnxruntime", "requests", "platformdirs", "pysbd"):
        sys.modules[name] = None  # importing any of them now raises ImportError
    import bytestts
    import bytestts.catalog.manifest, bytestts.text.tokenizer, bytestts.train.vits.config
    from bytestts import ConsentMissing, IntegrityError
    print("conversion-imports-ok", bytestts.__version__)
    """
)


def test_conversion_code_imports_without_the_runtime() -> None:
    proc = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter and a constant script
        [sys.executable, "-P", "-c", _NO_RUNTIME], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert f"conversion-imports-ok {bytestts.__version__}" in proc.stdout
