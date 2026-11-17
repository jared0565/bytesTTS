# SPDX-License-Identifier: MPL-2.0
"""torch tests: 1 thread (all-core load resets the development PC), and skipped cleanly without torch."""

try:
    import torch
except ImportError:  # the [train] extra is not installed
    collect_ignore_glob = ["*.py"]
else:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
