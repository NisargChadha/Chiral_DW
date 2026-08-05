#!/usr/bin/env python3
"""Recompute sewn AC topology, cG, and gaps from stored HF references."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.ac.nonideal import NonIdealACLLModel  # noqa: E402
from chiral_dw.ac.projected import build_ac_active_space  # noqa: E402
from chiral_dw.ac.response import (  # noqa: E402
    ACBandOverlapProvider,
    ACProjectorChernDiagnostics,
    ac_projector_chern_diagnostics,
    ac_reference_cherns_are_valid,
    k_theta_from_ac_projectors,
)
from chiral_dw.config import ACProjectedHFParams  # noqa: E402
from chiral_dw.continuum.models import MomentumGrid  # noqa: E402
from chiral_dw.continuum.references import symmetric_convex_path  # noqa: E402


SEWN_RESPONSE_METHOD = "ac_magnetic_bloch_sewn_overlap_v1"


@dataclass(frozen=True)
class StoredHFReferences:
    """Minimal convex-path interface backed by archived HF Hamiltonians."""

    H_vp_plus: np.ndarray
    H_vp_minus: np.ndarray
    H_ivc: np.ndarray
    n_occ_per_k: int

    @property
    def n_blocks(self) -> int:
        return int(self.H_vp_plus.shape[0])

    @property
    def dim(self) -> int:
        return int(self.H_vp_plus.shape[-1])


class StoredACResponseSummary(BaseModel):
    """Non-destructive sidecar summary for one archived AC sweep point."""

    model_config = ConfigDict(frozen=True)

    method: Literal["ac_magnetic_bloch_sewn_overlap_v1"]
    source_point_dir: str
    source_status: str
    hf_all_converged: bool
    n_k: int = Field(gt=0)
    n_ll: int = Field(gt=0)
    active_band: int = Field(ge=0)
    active_h0_reconstruction_residual: float = Field(ge=0.0)
    reference_topology_status: Literal[
        "valid",
        "numerically_unresolved",
        "symmetry_mismatch",
    ]
    reference_chern_symmetry_valid: bool
    reference_chern_numerically_resolved: bool
    reference_chern_valid: bool
    reference_chern_diagnostics: dict[str, ACProjectorChernDiagnostics]
    cG: float
    path_direct_gap_min: float
    path_indirect_gap_min: float


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hf_converged(summary: dict[str, Any]) -> bool:
    row = summary.get("row", {})
    if "hf_all_converged" in row:
        return bool(row["hf_all_converged"])
    diagnostics = summary.get("reference_diagnostics", [])
    return bool(diagnostics) and all(bool(item.get("converged")) for item in diagnostics)


def recompute_point(
    point_dir: Path,
    output_dir: Path,
    *,
    allow_nonconverged: bool = False,
) -> StoredACResponseSummary:
    """Write a sewn-response sidecar without modifying archived point files."""

    source = point_dir.resolve()
    summary_path = source / "point_summary.json"
    states_path = source / "reference_states.npz"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing stored point summary: {summary_path}")
    if not states_path.is_file():
        raise FileNotFoundError(f"missing stored HF references: {states_path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    source_summary = json.loads(summary_path.read_text())
    params = ACProjectedHFParams.model_validate(source_summary["params"])
    all_converged = _hf_converged(source_summary)
    if not all_converged and not allow_nonconverged:
        raise ValueError(
            "stored HF references did not all converge; pass --allow-nonconverged "
            "to compute a diagnostic response explicitly"
        )

    with np.load(states_path) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    required = {
        "active_h0",
        "vp_plus_P",
        "vp_plus_H_hf",
        "vp_minus_P",
        "vp_minus_H_hf",
        "ivc_P",
        "ivc_H_hf",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"stored HF archive is missing arrays: {sorted(missing)}")

    model = NonIdealACLLModel(params.ac)
    grid = MomentumGrid(params.grid.n_k)
    active, _bands = build_ac_active_space(
        model,
        grid,
        active_band=params.active_band,
        diagnostics_n_k=3,
    )
    h0_residual = float(np.max(np.abs(active.h0 - arrays["active_h0"])))
    if h0_residual > 1e-10:
        raise ValueError(
            "reconstructed AC active frame does not match the stored one-body Hamiltonian "
            f"(max residual {h0_residual:.3e})"
        )

    provider = ACBandOverlapProvider(
        model,
        active_band=params.active_band,
        active=active,
    )
    diagnostics = {
        "vp_plus": ac_projector_chern_diagnostics(
            provider, grid, arrays["vp_plus_P"]
        ),
        "vp_minus": ac_projector_chern_diagnostics(
            provider, grid, arrays["vp_minus_P"]
        ),
        "ivc": ac_projector_chern_diagnostics(provider, grid, arrays["ivc_P"]),
    }
    cherns = {name: item.chern for name, item in diagnostics.items()}
    symmetry_valid = ac_reference_cherns_are_valid(cherns)
    numerically_resolved = all(item.numerically_resolved for item in diagnostics.values())
    chern_valid = bool(symmetry_valid and numerically_resolved)
    topology_status = (
        "valid"
        if chern_valid
        else ("numerically_unresolved" if not numerically_resolved else "symmetry_mismatch")
    )

    refs = StoredHFReferences(
        H_vp_plus=arrays["vp_plus_H_hf"],
        H_vp_minus=arrays["vp_minus_H_hf"],
        H_ivc=arrays["ivc_H_hf"],
        n_occ_per_k=params.hf.n_occ_per_k,
    )
    theta_edges = np.linspace(
        params.response.theta_min,
        params.response.theta_max,
        params.response.n_theta + 1,
    )
    phi_nodes = np.arange(params.response.n_phi, dtype=float) * params.response.phi_step
    projectors, path_diagnostics = symmetric_convex_path(refs, theta_edges)
    projector_grid = projectors.reshape(
        len(theta_edges),
        grid.n_k,
        grid.n_k,
        active.dim,
        active.dim,
    )
    response = k_theta_from_ac_projectors(
        provider,
        projector_grid,
        theta_edges,
        phi_nodes,
    )
    direct_gap_min = float(min(item.direct_gap_min for item in path_diagnostics))
    indirect_gap_min = float(min(item.indirect_gap for item in path_diagnostics))
    result = StoredACResponseSummary(
        method=SEWN_RESPONSE_METHOD,
        source_point_dir=str(source),
        source_status=str(source_summary.get("row", {}).get("status", "unknown")),
        hf_all_converged=all_converged,
        n_k=grid.n_k,
        n_ll=params.ac.n_ll,
        active_band=params.active_band,
        active_h0_reconstruction_residual=h0_residual,
        reference_topology_status=topology_status,
        reference_chern_symmetry_valid=symmetry_valid,
        reference_chern_numerically_resolved=numerically_resolved,
        reference_chern_valid=chern_valid,
        reference_chern_diagnostics=diagnostics,
        cG=float(response.cG),
        path_direct_gap_min=direct_gap_min,
        path_indirect_gap_min=indirect_gap_min,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "summary.json", result.model_dump(mode="json"))
    _write_csv(
        output_dir / "chern_diagnostics.csv",
        [
            {"reference": name, **item.model_dump(mode="json")}
            for name, item in diagnostics.items()
        ],
    )
    _write_csv(
        output_dir / "path_theta_edges.csv",
        [
            {
                "theta": item.theta,
                "direct_gap_min": item.direct_gap_min,
                "indirect_gap": item.indirect_gap,
            }
            for item in path_diagnostics
        ],
    )
    _write_csv(
        output_dir / "response_K_theta.csv",
        [
            {"theta": float(theta), "K": float(value)}
            for theta, value in zip(response.theta, response.K, strict=True)
        ],
    )
    np.savez_compressed(
        output_dir / "response.npz",
        theta=np.asarray(response.theta, dtype=float),
        K=np.asarray(response.K, dtype=float),
        cG=np.asarray(response.cG, dtype=float),
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point_dir", type=Path, help="stored AC sweep point directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="sidecar directory (default: POINT_DIR/sewn_response_v1)",
    )
    parser.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help="compute a diagnostic response even when an archived HF reference did not converge",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="exit successfully when the sidecar directory already exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    point_dir = args.point_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else point_dir / "sewn_response_v1"
    )
    if output_dir.exists() and args.skip_existing:
        print(f"Skipping existing sewn response: {output_dir}")
        return 0
    result = recompute_point(
        point_dir,
        output_dir,
        allow_nonconverged=bool(args.allow_nonconverged),
    )
    print(
        f"Wrote {output_dir}: cG={result.cG:.12g}, "
        f"topology={result.reference_topology_status}, "
        f"indirect_gap_min={result.path_indirect_gap_min:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
