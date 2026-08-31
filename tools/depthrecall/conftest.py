# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make `depthrecall` importable without installing it.

The package is deliberately standalone -- it has its own `pyproject.toml` and does not import
`threedgrut` -- so it is not in the repo venv. Without this, the repo's own `pytest -q` from the
root collects these tests, fails to import them, and aborts the *entire* suite on a collection
error rather than skipping them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
