#!/usr/bin/env python3
"""Profile Taige finite-Q backend construction without running HF iterations."""

from __future__ import annotations

import argparse
from pathlib import Path

from chiral_dw.continuum.finite_q_memory_profile import (
    FINITE_Q_BUILD_VARIANTS,
    FiniteQBuildProfileParams,
    run_finite_q_build_profile_suite,
    run_finite_q_build_worker,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/taige_finite_q_build_memory",
    )
    parser.add_argument("--n-k-list", default="12,18,24")
    parser.add_argument(
        "--variants",
        default=",".join(FINITE_Q_BUILD_VARIANTS),
    )
    parser.add_argument("--theta-deg", type=float, default=3.5)
    parser.add_argument("--u-d", type=float, default=0.0)
    parser.add_argument("--plane-wave-shell", type=int, default=5)
    parser.add_argument("--n-bands", type=int, default=2)
    parser.add_argument("--n-active-bands-per-valley", type=int, default=2)
    parser.add_argument("--local-field-cutoff", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=50.0)
    parser.add_argument("--gate-distance-nm", type=float, default=30.0)
    parser.add_argument("--smear-length-nm", type=float, default=0.347)
    parser.add_argument("--vertex-workers", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--params-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    return parser


def _variants(value: str):
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(requested) - set(FINITE_Q_BUILD_VARIANTS)
    if unknown:
        raise ValueError(f"unknown finite-Q build variants: {sorted(unknown)}")
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.worker:
        if args.params_json is None or args.variant is None:
            parser.error("--worker requires --params-json and --variant")
        params = FiniteQBuildProfileParams.model_validate_json(args.params_json)
        variant = _variants(args.variant)[0]
        print(run_finite_q_build_worker(params, variant).model_dump_json())
        return 0

    n_k_values = tuple(int(part.strip()) for part in args.n_k_list.split(","))
    params = tuple(
        FiniteQBuildProfileParams(
            n_k=n_k,
            theta_deg=args.theta_deg,
            u_D=args.u_d,
            plane_wave_shell=args.plane_wave_shell,
            n_bands=args.n_bands,
            n_active_bands_per_valley=args.n_active_bands_per_valley,
            local_field_cutoff=args.local_field_cutoff,
            epsilon=args.epsilon,
            gate_distance_nm=args.gate_distance_nm,
            smear_length_nm=args.smear_length_nm,
            vertex_workers=args.vertex_workers,
        )
        for n_k in n_k_values
    )
    summary = run_finite_q_build_profile_suite(
        output_dir=Path(args.output_dir),
        script_path=Path(__file__).resolve(),
        params_by_n_k=params,
        variants=_variants(args.variants),
    )
    print(f"Wrote finite-Q build profile to {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
