#!/usr/bin/env python3
"""Merge per-mesh WSe2 IVC hysteresis outputs into finite-size diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from merge_taige_ivc_hysteresis_finite_size import main  # noqa: E402

DEFAULT_OUTPUT_ROOT = "results/wse2_ivc_hysteresis_finite_size_nk18_24_grid41"
DEFAULT_N_K_LIST = "18,19,20,21,22,23,24"


def _has_arg(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


if __name__ == "__main__":
    user_argv = list(sys.argv[1:])
    prefix: list[str] = []
    if not _has_arg(user_argv, "--output-root"):
        prefix.extend(["--output-root", DEFAULT_OUTPUT_ROOT])
    if not _has_arg(user_argv, "--n-k-list"):
        prefix.extend(["--n-k-list", DEFAULT_N_K_LIST])
    raise SystemExit(main([*prefix, *user_argv]))
