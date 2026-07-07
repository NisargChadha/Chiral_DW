#!/usr/bin/env python3
"""Merge WSe2 IVC hysteresis branch outputs into comparison tables."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from merge_taige_ivc_hysteresis_sweep import main  # noqa: E402

DEFAULT_OUTPUT_ROOT = "results/wse2_ivc_hysteresis_linear_interaction"


def _has_output_root(argv: list[str]) -> bool:
    return any(arg == "--output-root" or arg.startswith("--output-root=") for arg in argv)


if __name__ == "__main__":
    user_argv = list(sys.argv[1:])
    prefix = [] if _has_output_root(user_argv) else ["--output-root", DEFAULT_OUTPUT_ROOT]
    raise SystemExit(main([*prefix, *user_argv]))
