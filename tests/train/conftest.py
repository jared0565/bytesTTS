# SPDX-License-Identifier: MPL-2.0
"""torch tests: 1 thread (all-core load resets the development PC), and skipped cleanly without torch."""

import importlib.util

if importlib.util.find_spec("torch") is None:  # the [train] extra is not installed
    collect_ignore_glob = ["*.py"]
else:
    from helpers import pin_torch_threads

    pin_torch_threads()
