#!/usr/bin/env python3
"""Run Taige VP orbital-magnetization remote/HF convergence at one point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.config import (  # noqa: E402
    ContinuumGridParams,
    ContinuumHFParams,
    OrbitalMagnetizationParams,
    TaigeOrbitalMagnetizationWorkflowParams,
)
from chiral_dw.continuum.orbital_magnetization_workflow import (  # noqa: E402
    run_taige_orbital_magnetization_workflow,
)
from chiral_dw.continuum.taige import (  # noqa: E402
    taige_interaction_params,
    taige_model_params,
)


def _integer_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/taige_orbital_magnetization_theta3p7_uD0_nk18",
    )
    parser.add_argument("--theta-deg", type=float, default=3.7)
    parser.add_argument("--u-D", type=float, default=0.0)
    parser.add_argument("--n-k", type=int, default=18)
    parser.add_argument("--plane-wave-shell", type=int, default=5)
    parser.add_argument("--max-bands", type=int, default=8)
    parser.add_argument("--remote-cutoffs", type=_integer_tuple, default=tuple(range(7)))
    parser.add_argument("--hf-cutoffs", type=_integer_tuple, default=(2, 3, 4))
    parser.add_argument("--epsilon", type=float, default=16.7)
    parser.add_argument("--gate-distance-nm", type=float, default=30.0)
    parser.add_argument("--local-field-cutoff", type=int, default=4)
    parser.add_argument("--vertex-workers", type=int, default=1)
    parser.add_argument("--exchange-workers", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--no-k-resolved", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny noninteracting problem but the identical cache/artifact path.",
    )
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> TaigeOrbitalMagnetizationWorkflowParams:
    if args.smoke:
        n_k = 2
        shell = 1
        max_bands = 3
        remote_cutoffs = (0, 1)
        hf_cutoffs = (2,)
        interaction_scale = 0.0
        max_iter = 4
        benchmark_repeats = 2
        local_field_cutoff = 0
        q_mesh = "shell"
        q_shell = 0
    else:
        n_k = args.n_k
        shell = args.plane_wave_shell
        max_bands = args.max_bands
        remote_cutoffs = args.remote_cutoffs
        hf_cutoffs = args.hf_cutoffs
        interaction_scale = 1.0
        max_iter = args.max_iter
        benchmark_repeats = args.benchmark_repeats
        local_field_cutoff = args.local_field_cutoff
        q_mesh = "full"
        q_shell = 1
    model = taige_model_params(
        theta_deg=args.theta_deg,
        u_D=args.u_D,
        plane_wave_shell=shell,
        n_bands=max_bands,
        n_active_bands_per_valley=2,
    )
    interaction = taige_interaction_params(
        q_mesh=q_mesh,
        q_shell=q_shell,
        local_field_cutoff=local_field_cutoff,
        epsilon=args.epsilon,
        gate_distance_nm=args.gate_distance_nm,
        interaction_strength_scale=interaction_scale,
        vertex_workers=args.vertex_workers,
        exchange_workers=args.exchange_workers,
    )
    hf = ContinuumHFParams(
        n_occ_per_k=1,
        max_iter=max_iter,
        min_iter=0 if args.smoke else 2,
        mixing_method="linear" if args.smoke else "oda",
        mixing=1.0 if args.smoke else 0.45,
        tolerance=1e-9 if args.smoke else 1e-8,
        energy_tolerance=1e-11 if args.smoke else 1e-10,
    )
    orbital = OrbitalMagnetizationParams(
        remote_cutoffs_per_valley=remote_cutoffs,
        enlarged_hf_bands_per_valley=hf_cutoffs,
        benchmark_repeats=benchmark_repeats,
        store_k_resolved_terms=not args.no_k_resolved,
    )
    return TaigeOrbitalMagnetizationWorkflowParams(
        model=model,
        grid=ContinuumGridParams(n_k=n_k),
        interaction=interaction,
        hf=hf,
        orbital=orbital,
        output_dir=args.output_dir,
        reuse_completed_stages=not args.no_reuse,
    )


def main() -> int:
    params = build_params(parse_args())
    summary = run_taige_orbital_magnetization_workflow(params)
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if summary.manifest_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
