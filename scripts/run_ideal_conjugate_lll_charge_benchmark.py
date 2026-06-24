#!/usr/bin/env python3
"""Run the ideal opposite-Chern LLL real-space charge benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.cli import run_ideal_conjugate_lll_charge_console


if __name__ == "__main__":
    run_ideal_conjugate_lll_charge_console()
