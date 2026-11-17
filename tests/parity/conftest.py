# SPDX-License-Identifier: MPL-2.0
"""Parity tests compare the PyTorch port with the ONNX engine, so they need torch ([train] extra)."""

import importlib.util

collect_ignore_glob: list[str] = []
if importlib.util.find_spec("torch") is None:
    collect_ignore_glob.append("*.py")
else:
    from helpers import pin_torch_threads

    pin_torch_threads()
