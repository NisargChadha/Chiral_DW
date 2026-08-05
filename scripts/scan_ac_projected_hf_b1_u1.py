#!/usr/bin/env python3
"""Sweep finite-LL AC projected HF response over first-shell b1 and u1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.ac.projected import (  # noqa: E402
    AC_DENSITY_FORM_FACTOR_CONVENTION,
    build_ac_projected_bundle,
)
from chiral_dw.ac.response import (  # noqa: E402
    ACBandOverlapProvider,
    ac_projector_chern,
    ac_reference_cherns_are_valid,
    k_theta_from_ac_projectors,
)
from chiral_dw.config import (  # noqa: E402
    ACProjectedHFParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    FirstShellACParams,
    ResponseParams,
)
from chiral_dw.continuum import (  # noqa: E402
    build_symmetric_hf_references,
    order_diagnostics,
    reference_diagnostics,
    symmetric_convex_path,
)
from chiral_dw.continuum.momentum_channels import (  # noqa: E402
    c3_channel_index_map,
    c3_channel_value_residual,
    c3_spectrum_residual,
)

HBAR2_OVER_2ME_MEV_NM2 = 38.0998212


@dataclass(frozen=True)
class ACSweepPoint:
    b_index: int
    u_index: int
    b1: float
    u1: float

    @property
    def label(self) -> str:
        return f"b_{self.b_index:03d}_u_{self.u_index:03d}"

    def as_row(self) -> dict[str, Any]:
        return {
            "b_index": int(self.b_index),
            "u_index": int(self.u_index),
            "b1": float(self.b1),
            "u1": float(self.u1),
        }


class ACContinuumMatch(BaseModel):
    """Physical Taige-continuum scales used by a projected AC sweep."""

    model_config = ConfigDict(frozen=True)

    theta_deg: float = Field(gt=0.0)
    a0_angstrom: float = Field(gt=0.0)
    m_eff: float = Field(gt=0.0)
    interaction_normalization: Literal[
        "dimensionless_model",
        "omega_c_ratio",
    ]
    gate_distance_nm: float = Field(gt=0.0)
    interaction_multiplier: float = Field(ge=0.0)
    moire_length_nm: float = Field(gt=0.0)
    moire_cell_area_nm2: float = Field(gt=0.0)
    landau_level_spacing_mev: float = Field(gt=0.0)
    characteristic_coulomb_mev: float = Field(ge=0.0)
    characteristic_coulomb_to_ll_ratio: float = Field(ge=0.0)
    maximum_allowed_coulomb_to_ll_ratio: float = Field(gt=0.0)


def _continuum_match(args: argparse.Namespace) -> ACContinuumMatch:
    """Return and validate the physical continuum normalization for the AC run."""

    theta = np.deg2rad(float(args.continuum_theta_deg))
    a_m_nm = (
        float(args.continuum_a0_angstrom)
        / (2.0 * np.sin(0.5 * theta))
        / 10.0
    )
    area_nm2 = float(np.sqrt(3.0) * a_m_nm**2 / 2.0)
    l2_nm2 = area_nm2 / (2.0 * np.pi)
    omega_c_mev = float(
        2.0
        * HBAR2_OVER_2ME_MEV_NM2
        / (float(args.continuum_m_eff) * l2_nm2)
    )
    characteristic_coulomb_mev = float(args.v0) * omega_c_mev
    normalization = (
        "omega_c_ratio"
        if args.coulomb_kind == "dual_gate_omega_c"
        else "dimensionless_model"
    )
    ratio = characteristic_coulomb_mev / omega_c_mev
    maximum_ratio = float(args.max_coulomb_to_ll_ratio)
    if args.coulomb_kind == "dual_gate_omega_c" and ratio >= maximum_ratio:
        raise ValueError(
            "physical dual-gate interaction is too strong for the finite-LL projection: "
            f"E_C/(hbar*omega_c)={ratio:.6g} must be below {maximum_ratio:.6g}; "
            "reduce --v0 or raise the LL spacing through the continuum parameters"
        )

    derived_moire_length = a_m_nm
    derived_energy_unit = omega_c_mev
    if args.moire_length_nm is not None:
        if args.coulomb_kind == "dual_gate_omega_c" and not np.isclose(
            float(args.moire_length_nm), derived_moire_length, rtol=1e-8, atol=1e-10
        ):
            raise ValueError(
                "--moire-length-nm does not match the continuum-derived moire length; "
                "change --continuum-theta-deg/--continuum-a0-angstrom instead"
            )
        derived_moire_length = float(args.moire_length_nm)
    if args.energy_unit_mev is not None:
        if args.coulomb_kind == "dual_gate_omega_c" and not np.isclose(
            float(args.energy_unit_mev), derived_energy_unit, rtol=1e-8, atol=1e-10
        ):
            raise ValueError(
                "--energy-unit-mev does not match the continuum-derived LL spacing; "
                "change --continuum-theta-deg/--continuum-m-eff instead"
            )
        derived_energy_unit = float(args.energy_unit_mev)

    return ACContinuumMatch(
        theta_deg=float(args.continuum_theta_deg),
        a0_angstrom=float(args.continuum_a0_angstrom),
        m_eff=float(args.continuum_m_eff),
        interaction_normalization=normalization,
        gate_distance_nm=float(args.gate_distance_nm),
        interaction_multiplier=float(args.v0),
        moire_length_nm=derived_moire_length,
        moire_cell_area_nm2=area_nm2,
        landau_level_spacing_mev=derived_energy_unit,
        characteristic_coulomb_mev=characteristic_coulomb_mev,
        characteristic_coulomb_to_ll_ratio=ratio,
        maximum_allowed_coulomb_to_ll_ratio=maximum_ratio,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/ac_projected_hf_b1_u1_sweep")

    parser.add_argument("--b1", type=float, default=None, help="Run one explicit b1 value.")
    parser.add_argument("--u1", type=float, default=None, help="Run one explicit u1 value.")
    parser.add_argument("--b1-min", type=float, default=-0.3)
    parser.add_argument("--b1-max", type=float, default=0.3)
    parser.add_argument("--n-b1", type=int, default=11)
    parser.add_argument("--u1-min", type=float, default=-0.3)
    parser.add_argument("--u1-max", type=float, default=0.3)
    parser.add_argument("--n-u1", type=int, default=11)
    parser.add_argument("--task-id", type=int, default=None, help="SLURM-style flat grid index.")

    parser.add_argument("--n-ll", type=int, default=8)
    parser.add_argument("--active-band", type=int, default=0)
    parser.add_argument("--n-k", type=int, default=12)
    parser.add_argument("--band-diagnostics-n-k", type=int, default=9)

    parser.add_argument(
        "--coulomb-kind",
        choices=[
            "dimensionless_dual_gate",
            "dimensionless_screened",
            "dual_gate_omega_c",
        ],
        default="dual_gate_omega_c",
    )
    parser.add_argument("--interaction-strength-scale", "--v0", dest="v0", type=float, default=0.2)
    parser.add_argument("--dimensionless-gate-distance", "--gate-distance", dest="gate_distance", type=float, default=2.0)
    parser.add_argument("--q-mesh", choices=["shell", "full"], default="shell")
    parser.add_argument("--q-shell", type=int, default=1)
    parser.add_argument("--local-field-cutoff", type=int, default=1)
    parser.add_argument("--gate-distance-nm", type=float, default=30.0)
    parser.add_argument("--smear-length-nm", type=float, default=0.347)
    parser.add_argument("--omit-q0", action="store_true")
    parser.add_argument("--exchange-scale", type=float, default=1.0)
    parser.add_argument("--hartree-scale", type=float, default=1.0)
    parser.add_argument("--vertex-workers", type=int, default=1)
    parser.add_argument("--exchange-workers", type=int, default=1)
    parser.add_argument("--moire-length-nm", type=float, default=None)
    parser.add_argument("--energy-unit-mev", type=float, default=None)
    parser.add_argument("--continuum-theta-deg", type=float, default=3.5)
    parser.add_argument("--continuum-a0-angstrom", type=float, default=3.47)
    parser.add_argument("--continuum-m-eff", type=float, default=0.62)
    parser.add_argument(
        "--max-coulomb-to-ll-ratio",
        type=float,
        default=0.25,
        help="Require v0=E_C/(hbar*omega_c) to stay below this value.",
    )

    parser.add_argument("--n-occ-per-k", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--min-iter", type=int, default=2)
    parser.add_argument("--mixing-method", choices=["linear", "oda"], default="oda")
    parser.add_argument("--mixing", type=float, default=0.45)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--energy-tolerance", type=float, default=1e-10)
    parser.add_argument("--final-residual-tolerance", type=float, default=1e-7)
    parser.add_argument("--random-seed", type=int, default=1)

    parser.add_argument("--n-theta", type=int, default=81)
    parser.add_argument("--n-phi", type=int, default=5)
    parser.add_argument("--phi-step", type=float, default=0.2)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=pi)

    parser.add_argument(
        "--allow-nonconverged-response",
        action="store_true",
        help="Compute the convex response even when at least one HF reference did not converge.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write the selected scan plan without running HF.")
    parser.add_argument(
        "--no-write-plan",
        action="store_true",
        help="Do not rewrite the shared sweep plan (useful for parallel local task workers).",
    )
    parser.add_argument("--merge-only", action="store_true", help="Merge existing point summaries into sweep tables.")
    return parser


def _canonical_parameter(value: float) -> float:
    """Remove linspace roundoff before constructing the AC Hamiltonian."""

    return float(np.round(float(value), decimals=14))


def _linspace_points(args: argparse.Namespace) -> list[ACSweepPoint]:
    if (args.b1 is None) ^ (args.u1 is None):
        raise ValueError("--b1 and --u1 must be supplied together for an explicit single point")
    if args.b1 is not None and args.u1 is not None:
        return [
            ACSweepPoint(
                b_index=0,
                u_index=0,
                b1=_canonical_parameter(args.b1),
                u1=_canonical_parameter(args.u1),
            )
        ]
    if args.n_b1 < 1 or args.n_u1 < 1:
        raise ValueError("n-b1 and n-u1 must both be positive")
    b_values = [
        _canonical_parameter(value)
        for value in np.linspace(float(args.b1_min), float(args.b1_max), int(args.n_b1))
    ]
    u_values = [
        _canonical_parameter(value)
        for value in np.linspace(float(args.u1_min), float(args.u1_max), int(args.n_u1))
    ]
    return [
        ACSweepPoint(b_index=ib, u_index=iu, b1=float(b), u1=float(u))
        for ib, b in enumerate(b_values)
        for iu, u in enumerate(u_values)
    ]


def _selected_points(args: argparse.Namespace) -> list[ACSweepPoint]:
    points = _linspace_points(args)
    if args.task_id is None:
        return points
    task_id = int(args.task_id)
    if task_id < 0:
        raise ValueError("--task-id must be nonnegative")
    if task_id >= len(points):
        print(f"Task {task_id} is outside mesh size {len(points)}; exiting.")
        return []
    return [points[task_id]]


def _point_dir(output_root: Path, point: ACSweepPoint) -> Path:
    return output_root / "points" / point.label


def _params_for_point(
    args: argparse.Namespace,
    point: ACSweepPoint,
    point_dir: Path,
    continuum_match: ACContinuumMatch | None = None,
) -> ACProjectedHFParams:
    physical = continuum_match or _continuum_match(args)
    interaction = ContinuumInteractionParams(
        coulomb_kind=args.coulomb_kind,
        v0=float(args.v0),
        gate_distance=float(args.gate_distance),
        q_mesh=args.q_mesh,
        q_shell=int(args.q_shell),
        local_field_cutoff=int(args.local_field_cutoff),
        # The shared continuum schema retains epsilon for its physical-e^2
        # interaction.  AC sweep modes use v0 alone, so keep this inert field
        # fixed rather than exposing a second amplitude knob.
        epsilon=1.0,
        gate_distance_nm=float(args.gate_distance_nm),
        include_q0=not bool(args.omit_q0),
        smear_length_nm=float(args.smear_length_nm),
        exchange_scale=float(args.exchange_scale),
        hartree_scale=float(args.hartree_scale),
        vertex_workers=int(args.vertex_workers),
        exchange_workers=int(args.exchange_workers),
    )
    hf = ContinuumHFParams(
        n_occ_per_k=int(args.n_occ_per_k),
        max_iter=int(args.max_iter),
        min_iter=int(args.min_iter),
        mixing_method=args.mixing_method,
        mixing=float(args.mixing),
        tolerance=float(args.tolerance),
        energy_tolerance=float(args.energy_tolerance),
        final_residual_tolerance=float(args.final_residual_tolerance),
        random_seed=int(args.random_seed),
    )
    response = ResponseParams(
        n_theta=int(args.n_theta),
        n_phi=int(args.n_phi),
        phi_step=float(args.phi_step),
        theta_min=float(args.theta_min),
        theta_max=float(args.theta_max),
    )
    return ACProjectedHFParams(
        grid=ContinuumGridParams(n_k=int(args.n_k)),
        ac=FirstShellACParams(
            b1=float(point.b1),
            u1=float(point.u1),
            n_ll=int(args.n_ll),
        ),
        interaction=interaction,
        hf=hf,
        response=response,
        active_band=int(args.active_band),
        band_diagnostics_n_k=int(args.band_diagnostics_n_k),
        moire_length_nm=float(physical.moire_length_nm),
        energy_unit_mev=float(physical.landau_level_spacing_mev),
        output_dir=str(point_dir),
    )


def _reference_result_rows(point: ACSweepPoint, params: ACProjectedHFParams, bundle, refs, cherns: dict[str, float]) -> list[dict[str, Any]]:
    names = {
        "vp_plus": "VP+",
        "vp_minus": "VP-",
        "ivc": "IVC",
    }
    channel_diagnostics = reference_diagnostics(refs)
    rows: list[dict[str, Any]] = []
    norm = float(bundle.backend.n_blocks)
    energy_unit_mev = float(params.energy_unit_mev)
    for key, label in names.items():
        result = getattr(refs, key)
        diag = result.diagnostics
        order = order_diagnostics(result.P, bundle.active, n_occ_per_k=params.hf.n_occ_per_k)
        channels = channel_diagnostics[key]
        rows.append(
            {
                **point.as_row(),
                "reference": label,
                "energy": float(result.energy),
                "energy_over_omega_c": float(result.energy),
                "energy_mev": float(result.energy * energy_unit_mev),
                "energy_per_cell": float(result.energy / norm),
                "energy_per_cell_over_omega_c": float(result.energy / norm),
                "energy_per_cell_mev": float(result.energy / norm * energy_unit_mev),
                "converged": bool(result.converged),
                "n_iter": int(result.n_iter),
                "direct_gap_min": float(diag.direct_gap_min),
                "direct_gap_min_over_omega_c": float(diag.direct_gap_min),
                "direct_gap_min_mev": float(diag.direct_gap_min * energy_unit_mev),
                "indirect_gap": float(diag.indirect_gap),
                "indirect_gap_over_omega_c": float(diag.indirect_gap),
                "indirect_gap_mev": float(diag.indirect_gap * energy_unit_mev),
                "aufbau_residual_norm": float(diag.aufbau_residual_norm),
                "commutator_norm": float(diag.commutator_norm),
                "constraint_error": float(diag.constraint_error),
                "trace_error": float(diag.trace_error),
                "self_consistency_warning": bool(diag.self_consistency_warning),
                "ac_overlap_chern": float(cherns[key]),
                "Nz": float(order.Nz_block),
                "Nz_abs": float(order.Nz_abs),
                "IVC_amplitude": float(order.IVC_amplitude_block),
                "scalar_norm": float(channels.scalar_norm),
                "traceless_norm": float(channels.traceless_norm),
                "valley_diagonal_norm": float(channels.valley_diagonal_norm),
                "intervalley_norm": float(channels.intervalley_norm),
                "hermiticity_error": float(channels.hermiticity_error),
                "hf_spectrum_c3_residual": c3_spectrum_residual(
                    bundle.grid,
                    result.H_hf,
                ),
            }
        )
    return rows


def _path_rows(
    point: ACSweepPoint,
    params: ACProjectedHFParams,
    bundle,
    projectors: np.ndarray,
    path_diagnostics,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    norm = float(bundle.backend.n_blocks)
    energy_unit_mev = float(params.energy_unit_mev)
    for idx, (projector, diag) in enumerate(zip(projectors, path_diagnostics, strict=True)):
        energy = bundle.backend.energy(projector)
        rows.append(
            {
                **point.as_row(),
                "theta_index": int(idx),
                "theta": float(diag.theta),
                "theta_over_pi": float(diag.theta / pi),
                "w_vp_plus": float(diag.w_vp_plus),
                "w_vp_minus": float(diag.w_vp_minus),
                "w_ivc": float(diag.w_ivc),
                "direct_gap_min": float(diag.direct_gap_min),
                "direct_gap_min_over_omega_c": float(diag.direct_gap_min),
                "direct_gap_min_mev": float(diag.direct_gap_min * energy_unit_mev),
                "indirect_gap": float(diag.indirect_gap),
                "indirect_gap_over_omega_c": float(diag.indirect_gap),
                "indirect_gap_mev": float(diag.indirect_gap * energy_unit_mev),
                "projector_idempotency_error_fro": float(diag.projector_idempotency_error_fro),
                "projector_idempotency_error_max": float(diag.projector_idempotency_error_max),
                "energy_total_per_cell": float(energy.total / norm),
                "energy_total_per_cell_over_omega_c": float(energy.total / norm),
                "energy_total_per_cell_mev": float(
                    energy.total / norm * energy_unit_mev
                ),
                "energy_one_body_per_cell": float(energy.one_body / norm),
                "energy_one_body_per_cell_over_omega_c": float(
                    energy.one_body / norm
                ),
                "energy_one_body_per_cell_mev": float(
                    energy.one_body / norm * energy_unit_mev
                ),
                "energy_hartree_per_cell": float(energy.hartree / norm),
                "energy_hartree_per_cell_over_omega_c": float(
                    energy.hartree / norm
                ),
                "energy_hartree_per_cell_mev": float(
                    energy.hartree / norm * energy_unit_mev
                ),
                "energy_fock_per_cell": float(energy.fock / norm),
                "energy_fock_per_cell_over_omega_c": float(energy.fock / norm),
                "energy_fock_per_cell_mev": float(
                    energy.fock / norm * energy_unit_mev
                ),
            }
        )
    return rows


def _response_rows(point: ACSweepPoint, response) -> list[dict[str, Any]]:
    return [
        {
            **point.as_row(),
            "theta_center_index": int(idx),
            "theta": float(theta),
            "theta_over_pi": float(theta / pi),
            "K_theta": float(kappa),
            "cG": float(response.cG),
        }
        for idx, (theta, kappa) in enumerate(zip(response.theta, response.K, strict=True))
    ]


def _nan_response(params: ACProjectedHFParams):
    theta_centers = 0.5 * (
        np.linspace(params.response.theta_min, params.response.theta_max, params.response.n_theta + 1)[:-1]
        + np.linspace(params.response.theta_min, params.response.theta_max, params.response.n_theta + 1)[1:]
    )
    return type(
        "ResponseLike",
        (),
        {
            "theta": theta_centers,
            "K": np.full(theta_centers.shape, np.nan, dtype=float),
            "cG": float("nan"),
        },
    )()


def _load_point_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "row" not in data:
        raise ValueError(f"{path} does not contain a point-summary row")
    return dict(data["row"])


def _canonicalize_row_coordinates(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("b1", "u1"):
        if key in out and out[key] not in (None, ""):
            out[key] = _canonical_parameter(float(out[key]))
    return out


def _stack_point_table(output_root: Path, filename: str, output_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "points").glob(f"*/{filename}")):
        rows.extend(_canonicalize_row_coordinates(row) for row in _read_csv(path))
    _write_csv(output_root / output_name, rows)
    return rows


def _float_or_nan(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_sweep_arrays(output_root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    n_b = max(int(row["b_index"]) for row in rows) + 1
    n_u = max(int(row["u_index"]) for row in rows) + 1
    shape = (n_b, n_u)
    arrays: dict[str, np.ndarray] = {
        "b1": np.full(shape, np.nan, dtype=float),
        "u1": np.full(shape, np.nan, dtype=float),
        "cG": np.full(shape, np.nan, dtype=float),
        "bandwidth": np.full(shape, np.nan, dtype=float),
        "min_direct_gap": np.full(shape, np.nan, dtype=float),
        "min_direct_gap_mev": np.full(shape, np.nan, dtype=float),
        "interaction_gap_ratio": np.full(shape, np.nan, dtype=float),
        "characteristic_coulomb_to_active_band_gap_ratio": np.full(
            shape, np.nan, dtype=float
        ),
        "vp_plus_energy_per_cell": np.full(shape, np.nan, dtype=float),
        "vp_minus_energy_per_cell": np.full(shape, np.nan, dtype=float),
        "ivc_energy_per_cell": np.full(shape, np.nan, dtype=float),
        "ivc_minus_best_vp_energy_per_cell": np.full(shape, np.nan, dtype=float),
        "vp_plus_gap_mev": np.full(shape, np.nan, dtype=float),
        "vp_plus_indirect_gap_mev": np.full(shape, np.nan, dtype=float),
        "vp_minus_gap_mev": np.full(shape, np.nan, dtype=float),
        "vp_minus_indirect_gap_mev": np.full(shape, np.nan, dtype=float),
        "ivc_gap_mev": np.full(shape, np.nan, dtype=float),
        "ivc_indirect_gap_mev": np.full(shape, np.nan, dtype=float),
        "path_direct_gap_min_mev": np.full(shape, np.nan, dtype=float),
        "path_indirect_gap_min_mev": np.full(shape, np.nan, dtype=float),
        "chern_vp_plus": np.full(shape, np.nan, dtype=float),
        "chern_vp_minus": np.full(shape, np.nan, dtype=float),
        "chern_ivc": np.full(shape, np.nan, dtype=float),
        "hf_all_converged": np.zeros(shape, dtype=bool),
    }
    for row in rows:
        ib = int(row["b_index"])
        iu = int(row["u_index"])
        for key in arrays:
            if key == "hf_all_converged":
                arrays[key][ib, iu] = bool(row.get(key))
            else:
                arrays[key][ib, iu] = _float_or_nan(row, key)
    np.savez_compressed(output_root / "sweep_arrays.npz", **arrays)


def merge_point_summaries(output_root: Path) -> list[dict[str, Any]]:
    rows = [
        _canonicalize_row_coordinates(_load_point_summary(path))
        for path in sorted((output_root / "points").glob("*/point_summary.json"))
    ]
    rows.sort(key=lambda row: (int(row["b_index"]), int(row["u_index"])))
    _write_csv(output_root / "sweep.csv", rows)
    stacked_counts = {
        "reference_diagnostics": len(
            _stack_point_table(output_root, "reference_diagnostics.csv", "sweep_reference_diagnostics.csv")
        ),
        "hf_chern_numbers": len(
            _stack_point_table(output_root, "hf_chern_numbers.csv", "sweep_hf_chern_numbers.csv")
        ),
        "path_theta_edges": len(
            _stack_point_table(output_root, "path_theta_edges.csv", "sweep_path_theta_edges.csv")
        ),
        "response_K_theta": len(
            _stack_point_table(output_root, "response_K_theta.csv", "sweep_response_K_theta.csv")
        ),
    }
    _write_sweep_arrays(output_root, rows)
    _write_json(
        output_root / "sweep.json",
        {
            "rows": rows,
            "n_points": len(rows),
            "stacked_counts": stacked_counts,
            "tables": {
                "sweep_csv": str(output_root / "sweep.csv"),
                "sweep_arrays_npz": str(output_root / "sweep_arrays.npz"),
                "reference_diagnostics_csv": str(output_root / "sweep_reference_diagnostics.csv"),
                "hf_chern_numbers_csv": str(output_root / "sweep_hf_chern_numbers.csv"),
                "path_theta_edges_csv": str(output_root / "sweep_path_theta_edges.csv"),
                "response_K_theta_csv": str(output_root / "sweep_response_K_theta.csv"),
            },
        },
    )
    return rows


def _write_plan(output_root: Path, points: list[ACSweepPoint], args: argparse.Namespace) -> None:
    rows = [point.as_row() | {"point_dir": str(_point_dir(output_root, point))} for point in points]
    continuum_match = _continuum_match(args)
    _write_csv(output_root / "sweep_plan.csv", rows)
    _write_json(
        output_root / "sweep_plan.json",
        {
            "points": rows,
            "n_points": len(rows),
            "active_space_convention": "one active AC band per valley; default active_band=0 is the lowest band",
            "continuum_match": continuum_match.model_dump(mode="json"),
            "args": {
                key: value
                for key, value in vars(args).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        },
    )


def run_point(args: argparse.Namespace, output_root: Path, point: ACSweepPoint) -> dict[str, Any]:
    point_dir = _point_dir(output_root, point)
    point_summary = point_dir / "point_summary.json"
    if args.skip_existing and point_summary.exists():
        print(f"Skipping existing {point.label}: {point_summary}")
        return _load_point_summary(point_summary)

    continuum_match = _continuum_match(args)
    params = _params_for_point(args, point, point_dir, continuum_match)
    _write_json(point_dir / "point_params.json", params.model_dump(mode="json"))
    print(
        "Running AC projected HF "
        f"b1={point.b1:.8g} u1={point.u1:.8g} "
        f"n_k={params.grid.n_k} n_ll={params.ac.n_ll} active_band={params.active_band}"
    )
    start = time.perf_counter()

    bundle = build_ac_projected_bundle(params)
    refs = build_symmetric_hf_references(bundle, params.hf)
    provider = ACBandOverlapProvider(
        bundle.form_factors,
        active_band=params.active_band,
        active=bundle.active,
    )
    cherns = {
        "vp_plus": ac_projector_chern(provider, bundle.grid, refs.vp_plus.P),
        "vp_minus": ac_projector_chern(provider, bundle.grid, refs.vp_minus.P),
        "ivc": ac_projector_chern(provider, bundle.grid, refs.ivc.P),
    }
    reference_rows = _reference_result_rows(point, params, bundle, refs, cherns)
    _write_csv(point_dir / "reference_diagnostics.csv", reference_rows)
    _write_csv(
        point_dir / "hf_chern_numbers.csv",
        [
            {
                **point.as_row(),
                "reference": row["reference"],
                "chern": row["ac_overlap_chern"],
                "converged": row["converged"],
            }
            for row in reference_rows
        ],
    )

    all_converged = bool(refs.vp_plus.converged and refs.vp_minus.converged and refs.ivc.converged)
    reference_chern_valid = ac_reference_cherns_are_valid(cherns)
    response_status = "ok"
    projectors = None
    path_diagnostics = None
    if reference_chern_valid and (all_converged or args.allow_nonconverged_response):
        theta_edges = np.linspace(params.response.theta_min, params.response.theta_max, params.response.n_theta + 1)
        phi_nodes = np.arange(params.response.n_phi, dtype=float) * params.response.phi_step
        projectors, path_diagnostics = symmetric_convex_path(refs, theta_edges)
        projector_grid = projectors.reshape(
            len(theta_edges),
            bundle.grid.n_k,
            bundle.grid.n_k,
            bundle.active.dim,
            bundle.active.dim,
        )
        response = k_theta_from_ac_projectors(provider, projector_grid, theta_edges, phi_nodes)
        path_rows = _path_rows(point, params, bundle, projectors, path_diagnostics)
        response_rows = _response_rows(point, response)
    elif not reference_chern_valid:
        response_status = "skipped_invalid_reference_chern"
        response = _nan_response(params)
        path_rows = []
        response_rows = _response_rows(point, response)
    else:
        response_status = "skipped_nonconverged_hf"
        response = _nan_response(params)
        path_rows = []
        response_rows = _response_rows(point, response)

    _write_csv(point_dir / "path_theta_edges.csv", path_rows)
    _write_csv(point_dir / "response_K_theta.csv", response_rows)
    np.savez_compressed(
        point_dir / "response.npz",
        theta=np.asarray(response.theta, dtype=float),
        K=np.asarray(response.K, dtype=float),
        cG=np.asarray(response.cG, dtype=float),
    )
    np.savez_compressed(
        point_dir / "reference_states.npz",
        active_h0=np.asarray(bundle.active.h0, dtype=complex),
        k_fractional=np.asarray(bundle.bands.k_fractional, dtype=float),
        k_points=np.asarray(bundle.bands.k_points, dtype=float),
        vp_plus_P=np.asarray(refs.vp_plus.P, dtype=complex),
        vp_plus_H_hf=np.asarray(refs.vp_plus.H_hf, dtype=complex),
        vp_minus_P=np.asarray(refs.vp_minus.P, dtype=complex),
        vp_minus_H_hf=np.asarray(refs.vp_minus.H_hf, dtype=complex),
        ivc_P=np.asarray(refs.ivc.P, dtype=complex),
        ivc_H_hf=np.asarray(refs.ivc.H_hf, dtype=complex),
    )

    band_diag = bundle.bands.diagnostics if bundle.bands is not None else {}
    channel_mask = np.asarray(bundle.vertices.channel_in_disk, dtype=bool)
    channel_partner = c3_channel_index_map(
        bundle.grid,
        bundle.vertices.q_shifts,
        bundle.vertices.g_channels,
        channel_mask,
    )
    channel_c3_residual = c3_channel_value_residual(
        bundle.vertices.v_over_a,
        channel_partner,
        channel_mask,
    )
    interaction_scale = float(np.max(np.abs(bundle.vertices.v_over_a)))
    nonzero_channels = [
        abs(float(bundle.vertices.v_over_a[iq, ig]))
        for iq, q in enumerate(bundle.vertices.q_shifts)
        for ig, g in enumerate(bundle.vertices.g_channels)
        if not (q == (0, 0) and g == (0, 0))
    ]
    finite_q_interaction_scale = float(max(nonzero_channels)) if nonzero_channels else float("nan")
    min_direct_gap = float(band_diag.get("min_direct_gap", float("nan")))
    energy_unit_mev = float(params.energy_unit_mev)
    min_direct_gap_mev = min_direct_gap * energy_unit_mev
    characteristic_coulomb_to_active_band_gap_ratio = (
        float(continuum_match.characteristic_coulomb_mev)
        / max(min_direct_gap_mev, 1e-15)
    )
    active_band_projection_valid = bool(
        np.isfinite(characteristic_coulomb_to_active_band_gap_ratio)
        and characteristic_coulomb_to_active_band_gap_ratio < 1.0
    )
    vp_best_per_cell = min(float(refs.vp_plus.energy), float(refs.vp_minus.energy)) / float(bundle.backend.n_blocks)
    ivc_per_cell = float(refs.ivc.energy / bundle.backend.n_blocks)
    path_direct_gap_min = float("nan")
    path_indirect_gap_min = float("nan")
    path_idempotency_max = float("nan")
    if path_diagnostics is not None:
        path_direct_gap_min = float(
            np.min([row.direct_gap_min for row in path_diagnostics])
        )
        path_indirect_gap_min = float(
            np.min([row.indirect_gap for row in path_diagnostics])
        )
        path_idempotency_max = float(np.max([row.projector_idempotency_error_max for row in path_diagnostics]))
    elapsed = time.perf_counter() - start
    row = {
        **point.as_row(),
        "status": (
            "invalid_reference_chern"
            if not reference_chern_valid
            else ("ok" if all_converged else "nonconverged_hf")
        ),
        "response_status": response_status,
        "reference_chern_valid": reference_chern_valid,
        "elapsed_seconds": float(elapsed),
        "n_k": int(params.grid.n_k),
        "n_ll": int(params.ac.n_ll),
        "active_band": int(params.active_band),
        "n_active_bands_per_valley": int(bundle.active.n_active),
        "coulomb_kind": params.interaction.coulomb_kind,
        "interaction_normalization": continuum_match.interaction_normalization,
        "q_mesh": params.interaction.q_mesh,
        "v0_over_omega_c": float(params.interaction.v0),
        "interaction_multiplier": float(params.interaction.v0),
        "gate_distance": float(params.interaction.gate_distance),
        "gate_distance_nm": float(params.interaction.gate_distance_nm),
        "q_shell": int(params.interaction.q_shell),
        "local_field_cutoff": int(params.interaction.local_field_cutoff),
        "density_vertex_scheme": params.density_vertex_scheme,
        "density_form_factor_convention": AC_DENSITY_FORM_FACTOR_CONVENTION,
        "channel_candidate_count": int(channel_mask.size),
        "channel_active_count": int(np.count_nonzero(channel_mask)),
        "interaction_channel_c3_residual": channel_c3_residual,
        "bare_spectrum_c3_residual": c3_spectrum_residual(
            bundle.grid,
            bundle.active.h0,
        ),
        "vertex_workers": int(params.interaction.vertex_workers),
        "exchange_workers": int(params.interaction.exchange_workers),
        "moire_length_nm": float(params.moire_length_nm),
        "energy_unit": "hbar_omega_c",
        "landau_level_spacing_mev": float(params.energy_unit_mev),
        "characteristic_coulomb_mev": float(
            continuum_match.characteristic_coulomb_mev
        ),
        "characteristic_coulomb_to_ll_ratio": float(
            continuum_match.characteristic_coulomb_to_ll_ratio
        ),
        "interaction_scale": interaction_scale,
        "max_channel_weight_over_omega_c": interaction_scale,
        "finite_q_interaction_scale": finite_q_interaction_scale,
        "finite_q_max_channel_weight_over_omega_c": finite_q_interaction_scale,
        "interaction_gap_ratio": float(
            characteristic_coulomb_to_active_band_gap_ratio
        ),
        "characteristic_coulomb_to_active_band_gap_ratio": float(
            characteristic_coulomb_to_active_band_gap_ratio
        ),
        "active_band_projection_valid": active_band_projection_valid,
        "bandwidth": float(band_diag.get("bandwidth", float("nan"))),
        "bandwidth_over_omega_c": float(
            band_diag.get("bandwidth", float("nan"))
        ),
        "bandwidth_mev": float(
            band_diag.get("bandwidth", float("nan")) * energy_unit_mev
        ),
        "min_direct_gap": min_direct_gap,
        "min_direct_gap_over_omega_c": min_direct_gap,
        "min_direct_gap_mev": min_direct_gap_mev,
        "band_chern": float(band_diag.get("chern", float("nan"))),
        "berry_min": float(band_diag.get("berry_min", float("nan"))),
        "berry_max": float(band_diag.get("berry_max", float("nan"))),
        "berry_std": float(band_diag.get("berry_std", float("nan"))),
        "vp_plus_energy": float(refs.vp_plus.energy),
        "vp_plus_energy_over_omega_c": float(refs.vp_plus.energy),
        "vp_plus_energy_mev": float(refs.vp_plus.energy * energy_unit_mev),
        "vp_minus_energy": float(refs.vp_minus.energy),
        "vp_minus_energy_over_omega_c": float(refs.vp_minus.energy),
        "vp_minus_energy_mev": float(refs.vp_minus.energy * energy_unit_mev),
        "ivc_energy": float(refs.ivc.energy),
        "ivc_energy_over_omega_c": float(refs.ivc.energy),
        "ivc_energy_mev": float(refs.ivc.energy * energy_unit_mev),
        "vp_plus_energy_per_cell": float(refs.vp_plus.energy / bundle.backend.n_blocks),
        "vp_plus_energy_per_cell_over_omega_c": float(
            refs.vp_plus.energy / bundle.backend.n_blocks
        ),
        "vp_plus_energy_per_cell_mev": float(
            refs.vp_plus.energy / bundle.backend.n_blocks * energy_unit_mev
        ),
        "vp_minus_energy_per_cell": float(refs.vp_minus.energy / bundle.backend.n_blocks),
        "vp_minus_energy_per_cell_over_omega_c": float(
            refs.vp_minus.energy / bundle.backend.n_blocks
        ),
        "vp_minus_energy_per_cell_mev": float(
            refs.vp_minus.energy / bundle.backend.n_blocks * energy_unit_mev
        ),
        "ivc_energy_per_cell": ivc_per_cell,
        "ivc_energy_per_cell_over_omega_c": ivc_per_cell,
        "ivc_energy_per_cell_mev": float(ivc_per_cell * energy_unit_mev),
        "ivc_minus_best_vp_energy_per_cell": float(ivc_per_cell - vp_best_per_cell),
        "ivc_minus_best_vp_energy_per_cell_over_omega_c": float(
            ivc_per_cell - vp_best_per_cell
        ),
        "ivc_minus_best_vp_energy_per_cell_mev": float(
            (ivc_per_cell - vp_best_per_cell) * energy_unit_mev
        ),
        "vp_plus_converged": bool(refs.vp_plus.converged),
        "vp_minus_converged": bool(refs.vp_minus.converged),
        "ivc_converged": bool(refs.ivc.converged),
        "hf_all_converged": all_converged,
        "vp_plus_n_iter": int(refs.vp_plus.n_iter),
        "vp_minus_n_iter": int(refs.vp_minus.n_iter),
        "ivc_n_iter": int(refs.ivc.n_iter),
        "vp_plus_gap": float(refs.vp_plus.diagnostics.direct_gap_min),
        "vp_plus_gap_over_omega_c": float(
            refs.vp_plus.diagnostics.direct_gap_min
        ),
        "vp_plus_gap_mev": float(
            refs.vp_plus.diagnostics.direct_gap_min * energy_unit_mev
        ),
        "vp_minus_gap": float(refs.vp_minus.diagnostics.direct_gap_min),
        "vp_minus_gap_over_omega_c": float(
            refs.vp_minus.diagnostics.direct_gap_min
        ),
        "vp_minus_gap_mev": float(
            refs.vp_minus.diagnostics.direct_gap_min * energy_unit_mev
        ),
        "ivc_gap": float(refs.ivc.diagnostics.direct_gap_min),
        "ivc_gap_over_omega_c": float(refs.ivc.diagnostics.direct_gap_min),
        "ivc_gap_mev": float(
            refs.ivc.diagnostics.direct_gap_min * energy_unit_mev
        ),
        "vp_plus_indirect_gap_over_omega_c": float(
            refs.vp_plus.diagnostics.indirect_gap
        ),
        "vp_plus_indirect_gap_mev": float(
            refs.vp_plus.diagnostics.indirect_gap * energy_unit_mev
        ),
        "vp_minus_indirect_gap_over_omega_c": float(
            refs.vp_minus.diagnostics.indirect_gap
        ),
        "vp_minus_indirect_gap_mev": float(
            refs.vp_minus.diagnostics.indirect_gap * energy_unit_mev
        ),
        "ivc_indirect_gap_over_omega_c": float(
            refs.ivc.diagnostics.indirect_gap
        ),
        "ivc_indirect_gap_mev": float(
            refs.ivc.diagnostics.indirect_gap * energy_unit_mev
        ),
        "vp_plus_residual": float(refs.vp_plus.diagnostics.aufbau_residual_norm),
        "vp_minus_residual": float(refs.vp_minus.diagnostics.aufbau_residual_norm),
        "ivc_residual": float(refs.ivc.diagnostics.aufbau_residual_norm),
        "vp_plus_hf_spectrum_c3_residual": c3_spectrum_residual(
            bundle.grid,
            refs.vp_plus.H_hf,
        ),
        "vp_minus_hf_spectrum_c3_residual": c3_spectrum_residual(
            bundle.grid,
            refs.vp_minus.H_hf,
        ),
        "ivc_hf_spectrum_c3_residual": c3_spectrum_residual(
            bundle.grid,
            refs.ivc.H_hf,
        ),
        "chern_vp_plus": float(cherns["vp_plus"]),
        "chern_vp_minus": float(cherns["vp_minus"]),
        "chern_ivc": float(cherns["ivc"]),
        "cG": float(response.cG),
        "K_min": float(np.nanmin(response.K)) if np.any(np.isfinite(response.K)) else float("nan"),
        "K_max": float(np.nanmax(response.K)) if np.any(np.isfinite(response.K)) else float("nan"),
        "path_gap_min": path_direct_gap_min,
        "path_direct_gap_min_over_omega_c": path_direct_gap_min,
        "path_direct_gap_min_mev": float(path_direct_gap_min * energy_unit_mev),
        "path_indirect_gap_min_over_omega_c": path_indirect_gap_min,
        "path_indirect_gap_min_mev": float(
            path_indirect_gap_min * energy_unit_mev
        ),
        "path_projector_idempotency_error_max": path_idempotency_max,
        "point_dir": str(point_dir),
    }
    _write_json(
        point_summary,
        {
            "point": point.as_row(),
            "row": row,
            "params": params.model_dump(mode="json"),
            "continuum_match": continuum_match.model_dump(mode="json"),
            "band_diagnostics": band_diag,
            "reference_diagnostics": reference_rows,
            "active_space_convention": "one active AC band per valley; active_band selects the finite-LL band before HF",
            "density_form_factor_convention": AC_DENSITY_FORM_FACTOR_CONVENTION,
        },
    )
    print(
        f"Finished {point.label}: cG={row['cG']:.12g} "
        f"C(VP+,VP-,IVC)=({row['chern_vp_plus']:.6g},{row['chern_vp_minus']:.6g},{row['chern_ivc']:.6g}) "
        f"status={row['status']} elapsed={elapsed:.1f}s"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        rows = merge_point_summaries(output_root)
        print(f"Merged {len(rows)} point summaries into {output_root / 'sweep.csv'}")
        return 0

    points = _selected_points(args)
    if not args.no_write_plan:
        _write_plan(output_root, points, args)
    if args.dry_run:
        print(f"Wrote dry-run plan with {len(points)} selected point(s) to {output_root}")
        return 0
    if not points:
        return 0

    _ = [run_point(args, output_root, point) for point in points]
    if args.task_id is None:
        rows = merge_point_summaries(output_root)
        print(f"Wrote serial sweep summary with {len(rows)} points to {output_root / 'sweep.csv'}")
    else:
        print("Single task complete. Run with --merge-only after the array finishes to build sweep.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
