#!/usr/bin/env python3
"""Run local Taige HF memory-backend benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from chiral_dw.continuum.memory_benchmark import (
    TaigeMemoryBenchmarkInput,
    parse_backend_variants,
    parse_n_k_list,
    run_taige_memory_benchmark_suite,
    run_taige_memory_benchmark_worker,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/taige_memory_benchmarks")
    parser.add_argument("--n-k-list", default="6,8,10,12")
    parser.add_argument(
        "--variants",
        default="baseline,hartree_only,fused,compact,fused_compact,packed,matrix_free",
        help="Comma-separated backend variants, or 'all'.",
    )
    parser.add_argument("--u-d", type=float, default=0.0)
    parser.add_argument("--theta-deg", type=float, default=3.5)
    parser.add_argument("--plane-wave-shell", type=int, default=5)
    parser.add_argument("--n-bands", type=int, default=2)
    parser.add_argument("--n-active-bands-per-valley", type=int, default=2)
    parser.add_argument("--q-mesh", choices=("shell", "full"), default="full")
    parser.add_argument("--q-shell", type=int, default=0)
    parser.add_argument("--local-field-cutoff", type=int, default=4)
    parser.add_argument("--vertex-workers", type=int, default=1)
    parser.add_argument("--exchange-workers", type=int, default=1)
    parser.add_argument("--fock-repeats", type=int, default=25)
    parser.add_argument("--max-rss-gb", type=float, default=None)
    parser.add_argument("--run-hf-smoke", action="store_true")
    parser.add_argument("--hf-max-iter", type=int, default=6)
    parser.add_argument(
        "--no-subprocess",
        action="store_true",
        help="Debug/testing mode; by default each variant runs in a fresh subprocess.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--params-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reference-input", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reference-output", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.worker:
        if args.params_json is None or args.variant is None:
            parser.error("--worker requires --params-json and --variant")
        params = TaigeMemoryBenchmarkInput.model_validate_json(args.params_json)
        result = run_taige_memory_benchmark_worker(
            params=params,
            variant=parse_backend_variants(args.variant)[0],
            reference_input=None if args.reference_input is None else Path(args.reference_input),
            reference_output=None if args.reference_output is None else Path(args.reference_output),
        )
        print(result.model_dump_json())
        return 0

    base_params = TaigeMemoryBenchmarkInput(
        u_D=args.u_d,
        theta_deg=args.theta_deg,
        plane_wave_shell=args.plane_wave_shell,
        n_bands=args.n_bands,
        n_active_bands_per_valley=args.n_active_bands_per_valley,
        q_mesh=args.q_mesh,
        q_shell=args.q_shell,
        local_field_cutoff=args.local_field_cutoff,
        vertex_workers=args.vertex_workers,
        exchange_workers=args.exchange_workers,
        fock_repeats=args.fock_repeats,
        run_hf_smoke=bool(args.run_hf_smoke),
        hf_max_iter=args.hf_max_iter,
        max_rss_gb=args.max_rss_gb,
    )
    summary = run_taige_memory_benchmark_suite(
        output_dir=Path(args.output_root),
        script_path=Path(__file__).resolve(),
        base_params=base_params,
        n_k_list=parse_n_k_list(args.n_k_list),
        variants=parse_backend_variants(args.variants),
        use_subprocess=not bool(args.no_subprocess),
    )
    print(f"Wrote Taige memory benchmark outputs to {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
