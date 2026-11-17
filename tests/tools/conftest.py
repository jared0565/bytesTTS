# SPDX-License-Identifier: MPL-2.0
"""tests/tools: coqui_keys and the archive checks need no torch; packaging and convert_state do ([train] extra)."""

import importlib.util

collect_ignore_glob: list[str] = []
if importlib.util.find_spec("torch") is None:
    collect_ignore_glob += ["test_package_voice.py", "test_convert_state.py"]
else:
    from helpers import pin_torch_threads

    pin_torch_threads()
