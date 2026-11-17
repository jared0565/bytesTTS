# SPDX-License-Identifier: MPL-2.0
"""Thread caps for EVERY test run on the development PC: mine, aramid's gate, and aramid's drain,
whose mutation loop re-runs this suite unpinned.

Sustained all-core ONNX Runtime load hard-reset the development PC twice (plan, Global Constraints,
"This PC"). So:
- the BLAS/OpenMP pools are capped before any test module imports numpy or torch;
- an ONNX Runtime session created without `intra_op_num_threads == 1` FAILS the test. It fails
  rather than being capped quietly, so a missing `threads=1` shows up and a test about threads
  can't pass vacuously.

Nothing to import from here: shared helpers live in tests/helpers.py.
"""

import os
from collections.abc import Iterator
from typing import Any

import pytest

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


@pytest.fixture(autouse=True)
def _onnxruntime_sessions_must_be_single_threaded(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import onnxruntime as ort

    real = ort.InferenceSession

    def guarded(path_or_bytes: Any, sess_options: Any = None, *args: Any, **kwargs: Any) -> Any:
        if sess_options is None or sess_options.intra_op_num_threads != 1:
            pytest.fail(
                "ONNX Runtime session without intra_op_num_threads=1 (pass threads=1 / TEST_THREADS): "
                "all-core ORT load resets the development PC"
            )
        return real(path_or_bytes, sess_options, *args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", guarded)
    yield
