"""Console entry points for chiral domain-wall workflows."""

from __future__ import annotations

import argparse

from chiral_dw.ac.workflow import run_ac_cg_workflow
from chiral_dw.config import (
    ACResponseWorkflowParams,
    DomainWallParams,
    FirstShellACParams,
    GatedInteractionParams,
    MomentumGridParams,
    ResponseParams,
    SourceInterpolationParams,
)


def _ac_cg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a nonideal AC cG workflow.")
    parser.add_argument("--output-dir", default="results/ac_cg")
    parser.add_argument("--b1", type=float, default=0.0)
    parser.add_argument("--u1", type=float, default=0.0)
    parser.add_argument("--b1-c3", type=float, default=0.0)
    parser.add_argument("--u1-c3", type=float, default=0.0)
    parser.add_argument("--n-ll", type=int, default=3)
    parser.add_argument("--n-k", type=int, default=5)
    parser.add_argument("--n-theta", type=int, default=21)
    parser.add_argument("--source-scale", type=float, default=1.0)
    parser.add_argument("--interaction-shell", type=int, default=1)
    parser.add_argument("--gate-distance", type=float, default=2.0)
    parser.add_argument("--v0", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--winding", type=int, default=1)
    parser.add_argument("--plots", action="store_true")
    return parser


def run_ac_cg_console() -> None:
    args = _ac_cg_parser().parse_args()
    params = ACResponseWorkflowParams(
        grid=MomentumGridParams(n_k=args.n_k),
        ac=FirstShellACParams(
            b1=args.b1,
            u1=args.u1,
            b1_c3=args.b1_c3,
            u1_c3=args.u1_c3,
            n_ll=args.n_ll,
        ),
        response=ResponseParams(n_theta=args.n_theta),
        source=SourceInterpolationParams(source_scale=args.source_scale),
        interaction=GatedInteractionParams(
            v0=args.v0,
            gate_distance=args.gate_distance,
            interaction_shell=args.interaction_shell,
        ),
        domain_wall=DomainWallParams(radius=args.radius, width=args.width, winding=args.winding),
        output_dir=args.output_dir,
    )
    result = run_ac_cg_workflow(params, write_outputs=True, write_plots=args.plots)
    print(f"cG = {result.response.cG:.12g}")
    print(f"output_dir = {params.output_dir}")
