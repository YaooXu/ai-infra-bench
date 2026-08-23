#!/usr/bin/env python3
"""Trusted pytest entrypoint that resolves vLLM from the candidate workspace."""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Import the verifier-owned framework before exposing candidate source on
# sys.path, preventing a candidate pytest.py from replacing the test runner.
import pytest


workspace = Path(os.environ.get("BENCH_WORKSPACE", "/app")).resolve()
sys.path.insert(0, str(workspace))
raise SystemExit(pytest.main(sys.argv[1:]))
