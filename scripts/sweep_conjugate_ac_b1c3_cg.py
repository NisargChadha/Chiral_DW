#!/usr/bin/env python3
"""Run the old-compatible finite-LL magnetic C3 conjugate-AC bias sweep."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.ac.bias_sweep import run_b1c3_sweep_console


if __name__ == "__main__":
    run_b1c3_sweep_console()
