#!/usr/bin/env python3
"""Run one WSe2 Q=0 IVC hysteresis linecut branch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_taige_ivc_hysteresis_linecut import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--material", "wse2", *sys.argv[1:]]))
