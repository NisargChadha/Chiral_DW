"""Console entry points for chiral domain-wall workflows."""

from __future__ import annotations

import argparse

from chiral_dw.ac.workflow import run_ac_cg_workflow
from chiral_dw.config import (
    ACResponseWorkflowParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
    ContinuumWorkflowParams,
    DomainWallParams,
    FirstShellACParams,
    GatedInteractionParams,
    IdealConjugateLLLChargeBenchmarkParams,
    MomentumGridParams,
    QHFMChargeBenchmarkParams,
    RealSpaceGridParams,
    ResponseParams,
    SkyrmionTextureParams,
    SourceInterpolationParams,
)
from chiral_dw.continuum.workflow import run_continuum_symmetric_hf_workflow
from chiral_dw.ideal_conjugate_lll import run_ideal_conjugate_lll_charge_benchmark
from chiral_dw.qhfm_benchmark import run_qhfm_charge_benchmark


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


def _continuum_symmetric_hf_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the native continuum symmetric-HF workflow.")
    parser.add_argument("--output-dir", default="results/continuum_symmetric_hf")
    parser.add_argument("--n-k", type=int, default=5)
    parser.add_argument("--n-theta", type=int, default=21)
    parser.add_argument("--v0", type=float, default=1.0)
    parser.add_argument("--gate-distance", type=float, default=2.0)
    parser.add_argument("--q-shell", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--mixing", type=float, default=0.45)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--displacement-mev", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--winding", type=int, default=1)
    return parser


def run_continuum_symmetric_hf_console() -> None:
    args = _continuum_symmetric_hf_parser().parse_args()
    params = ContinuumWorkflowParams(
        grid=ContinuumGridParams(n_k=args.n_k),
        model=ContinuumModelParams(displacement_mev=args.displacement_mev),
        interaction=ContinuumInteractionParams(
            v0=args.v0,
            gate_distance=args.gate_distance,
            q_shell=args.q_shell,
        ),
        hf=ContinuumHFParams(
            max_iter=args.max_iter,
            mixing=args.mixing,
            tolerance=args.tolerance,
        ),
        response=ResponseParams(n_theta=args.n_theta),
        domain_wall=DomainWallParams(radius=args.radius, width=args.width, winding=args.winding),
        output_dir=args.output_dir,
    )
    result = run_continuum_symmetric_hf_workflow(params, write_outputs=True)
    print(f"cG = {result.response.cG:.12g}")
    print(f"vp_plus_idempotency = {result.reference_summary['vp_plus']['idempotency_error_fro']:.3e}")
    print(f"vp_minus_idempotency = {result.reference_summary['vp_minus']['idempotency_error_fro']:.3e}")
    print(f"ivc_idempotency = {result.reference_summary['ivc']['idempotency_error_fro']:.3e}")
    print(f"output_dir = {params.output_dir}")


def _qhfm_charge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the same-Chern QHFM charge benchmark.")
    parser.add_argument("--output-dir", default="results/qhfm_charge_benchmark")
    parser.add_argument("--b1", type=float, default=0.2)
    parser.add_argument("--u1", type=float, default=0.1)
    parser.add_argument("--b1-c3", type=float, default=0.0)
    parser.add_argument("--u1-c3", type=float, default=0.0)
    parser.add_argument("--n-ll", type=int, default=5)
    parser.add_argument("--n-k", type=int, default=7)
    parser.add_argument("--n-r", type=int, default=9)
    parser.add_argument("--active-band", type=int, default=0)
    parser.add_argument("--skyrmion-mass", type=float, default=0.5)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--no-curvature-npz", action="store_true")
    return parser


def run_qhfm_charge_console() -> None:
    args = _qhfm_charge_parser().parse_args()
    params = QHFMChargeBenchmarkParams(
        grid=MomentumGridParams(n_k=args.n_k),
        real_space=RealSpaceGridParams(n_r=args.n_r),
        ac=FirstShellACParams(
            b1=args.b1,
            u1=args.u1,
            b1_c3=args.b1_c3,
            u1_c3=args.u1_c3,
            n_ll=args.n_ll,
        ),
        skyrmion=SkyrmionTextureParams(mass=args.skyrmion_mass),
        active_band=args.active_band,
        output_dir=args.output_dir,
        write_curvature_npz=not args.no_curvature_npz,
    )
    result = run_qhfm_charge_benchmark(params, write_outputs=True, write_plots=args.plots)
    print(f"orbital_chern = {result.summary.orbital_chern:.12g}")
    print(f"mixed_curvature_max = {result.summary.mixed_curvature_max:.3e}")
    print(f"charge_error_max = {result.summary.charge_error_max:.3e}")
    print(f"integrated_charge = {result.summary.integrated_charge:.12g}")
    print(f"integrated_skyrmion_charge = {result.summary.integrated_skyrmion_charge:.12g}")
    print(f"valid_charge_normalization = {result.summary.valid_charge_normalization}")
    print(f"output_dir = {params.output_dir}")


def _ideal_conjugate_lll_charge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ideal conjugate LLL charge benchmark.")
    parser.add_argument("--output-dir", default="results/ideal_conjugate_lll_charge")
    parser.add_argument("--n-k", type=int, default=7)
    parser.add_argument("--n-r", type=int, default=41)
    parser.add_argument("--radius-lb", type=float, default=10.0)
    parser.add_argument("--width-lb", type=float, default=3.5)
    parser.add_argument("--patch-length-lb", type=float, default=56.0)
    parser.add_argument("--winding", type=int, default=1)
    parser.add_argument("--helicity", type=float, default=0.0)
    parser.add_argument("--m0", type=float, default=1.0)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--no-curvature-npz", action="store_true")
    return parser


def run_ideal_conjugate_lll_charge_console() -> None:
    args = _ideal_conjugate_lll_charge_parser().parse_args()
    params = IdealConjugateLLLChargeBenchmarkParams(
        grid=MomentumGridParams(n_k=args.n_k),
        real_space=RealSpaceGridParams(n_r=args.n_r),
        radius_lB=args.radius_lb,
        width_lB=args.width_lb,
        patch_length_lB=args.patch_length_lb,
        winding=args.winding,
        helicity=args.helicity,
        m0=args.m0,
        output_dir=args.output_dir,
        write_curvature_npz=not args.no_curvature_npz,
    )
    result = run_ideal_conjugate_lll_charge_benchmark(
        params,
        write_outputs=True,
        write_plots=args.plots,
    )
    print(f"up_chern = {result.summary.up_chern:.12g}")
    print(f"down_chern = {result.summary.down_chern:.12g}")
    print(f"charge_error_max = {result.summary.charge_error_max:.3e}")
    print(f"charge_error_rms = {result.summary.charge_error_rms:.3e}")
    print(f"integrated_charge = {result.summary.integrated_charge:.12g}")
    print(f"integrated_analytic_charge = {result.summary.integrated_analytic_charge:.12g}")
    print(f"valid_analytic_charge = {result.summary.valid_analytic_charge}")
    print(f"output_dir = {params.output_dir}")
