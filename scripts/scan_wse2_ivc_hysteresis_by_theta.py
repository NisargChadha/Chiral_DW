#!/usr/bin/env python3
"""Compatibility wrapper for fixed-theta WSe2 IVC displacement hysteresis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_taige_ivc_hysteresis_linecut import main  # noqa: E402


if __name__ == "__main__":
    argv = ["--material", "wse2", "--sweep-axis", "u_D", *sys.argv[1:]]
    raise SystemExit(main(argv))
