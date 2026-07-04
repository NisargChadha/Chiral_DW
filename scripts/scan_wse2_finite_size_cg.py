#!/usr/bin/env python3
"""Finite-size WSe2 Taige-continuum symmetric-HF cG sweep."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_taige_finite_size_cg import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--material", "wse2", *sys.argv[1:]]))
